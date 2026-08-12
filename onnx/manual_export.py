"""Memory-bounded ONNX graph builder for BS-RoFormer.

TorchScript tracing materializes the full attention tensors while exporting a
1,722-frame model and can exceed workstation memory.  This builder emits the
same standard ONNX operators directly from the loaded PyTorch weights, so
export memory is proportional to graph metadata rather than the attention
working set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .config import SeparationConfig
    from .model import SpectralCore, load_reference_model, set_math_attention
except ImportError:  # direct script imports from the onnx directory
    from config import SeparationConfig  # type: ignore[no-redef]
    from model import SpectralCore, load_reference_model, set_math_attention  # type: ignore[no-redef]


@dataclass
class GraphBuilder:
    helper: Any
    tensor_proto: Any
    nodes: list[Any]
    initializers: list[Any]
    float_dtype: Any = np.dtype(np.float32)
    counter: int = 0

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def initializer(self, prefix: str, value: np.ndarray | float | int, dtype: Any | None = None) -> str:
        name = self.name(prefix)
        array = np.asarray(value)
        if array.dtype.kind in "iu":
            array = array.astype(np.int64)
        else:
            array = array.astype(self.float_dtype if dtype is None else dtype)
        self.initializers.append(
            __import__("onnx").numpy_helper.from_array(array, name=name)
        )
        return name

    def node(self, op: str, inputs: list[str], outputs: list[str] | None = None, prefix: str | None = None, **attrs: Any) -> str | list[str]:
        output = outputs or [self.name(prefix or op.lower())]
        self.nodes.append(self.helper.make_node(op, inputs, output, **attrs))
        return output if outputs is not None else output[0]

    def constant(self, value: np.ndarray | float | int, prefix: str = "const") -> str:
        return self.initializer(prefix, value)


def _linear(builder: GraphBuilder, value: str, module: Any, prefix: str) -> str:
    if np.dtype(builder.float_dtype) == np.dtype(np.float16):
        value = builder.node("Cast", [value], to=builder.tensor_proto.FLOAT16, prefix=f"{prefix}_input_fp16")
    weight = module.weight.detach().cpu().numpy().astype(np.float32).T
    result = builder.node("MatMul", [value, builder.initializer(f"{prefix}_weight", weight)])
    if module.bias is not None:
        result = builder.node(
            "Add",
            [result, builder.initializer(f"{prefix}_bias", module.bias.detach().cpu().numpy())],
        )
    return result


def _rms_norm(builder: GraphBuilder, value: str, module: Any, prefix: str) -> str:
    # torch.cuda.amp.autocast leaves F.normalize in FP32.  Preserve that
    # behavior for FP16 graphs; otherwise tiny errors compound through the 16
    # transformer blocks and materially change the mask.
    norm_dtype = np.dtype(np.float32) if np.dtype(builder.float_dtype) == np.dtype(np.float16) else builder.float_dtype
    if norm_dtype == np.dtype(np.float32) and np.dtype(builder.float_dtype) == np.dtype(np.float16):
        value = builder.node("Cast", [value], to=builder.tensor_proto.FLOAT, prefix=f"{prefix}_input_fp32")
    squared = builder.node("Mul", [value, value], prefix=f"{prefix}_square")
    summed = builder.node("ReduceSum", [squared, builder.initializer(f"{prefix}_axes", np.array([-1], dtype=np.int64))], keepdims=1)
    epsilon = builder.initializer(f"{prefix}_epsilon", np.asarray(1.0e-12, dtype=np.float32), dtype=np.float32 if norm_dtype == np.dtype(np.float32) else None)
    norm = builder.node("Sqrt", [builder.node("Add", [summed, epsilon])])
    normalized = builder.node("Div", [value, norm])
    scale = builder.initializer(f"{prefix}_scale", np.asarray(module.scale, dtype=np.float32), dtype=np.float32 if norm_dtype == np.dtype(np.float32) else None)
    normalized = builder.node("Mul", [normalized, scale])
    gamma = builder.initializer(f"{prefix}_gamma", module.gamma.detach().cpu().numpy(), dtype=np.float32 if norm_dtype == np.dtype(np.float32) else None)
    return builder.node("Mul", [normalized, gamma])


def _rotary(builder: GraphBuilder, value: str, sequence_length: int, dim: int, prefix: str) -> str:
    # rotary_embedding_torch repeats each frequency twice, then rotates
    # contiguous pairs: [x0, x1] -> [-x1, x0].
    frequencies = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    angles = np.arange(sequence_length, dtype=np.float32)[:, None] * frequencies[None, :]
    angles = np.repeat(angles, 2, axis=1)
    cos = builder.initializer(f"{prefix}_cos", np.cos(angles)[None, None, :, :])
    sin = builder.initializer(f"{prefix}_sin", np.sin(angles)[None, None, :, :])
    rotated = builder.node("Mul", [value, cos])
    reshaped = builder.node("Reshape", [value, builder.initializer(f"{prefix}_pair_shape", np.array([0, 0, 0, -1, 2], dtype=np.int64))])
    negated = builder.node("Neg", [builder.node("Gather", [reshaped, builder.constant(np.array(1, dtype=np.int64), f"{prefix}_one")], axis=4)])
    positive = builder.node("Gather", [reshaped, builder.constant(np.array(0, dtype=np.int64), f"{prefix}_zero")], axis=4)
    # Gather removes the pair axis.  Reinsert it before concatenating so the
    # result is [-x1, x0, -x3, x2, ...], rather than all negatives followed
    # by all positives.
    negated = builder.node("Unsqueeze", [negated, builder.initializer(f"{prefix}_neg_axis", np.array([4], dtype=np.int64))])
    positive = builder.node("Unsqueeze", [positive, builder.initializer(f"{prefix}_pos_axis", np.array([4], dtype=np.int64))])
    rotated_half = builder.node("Concat", [negated, positive], axis=4)
    rotated_half = builder.node("Reshape", [rotated_half, builder.initializer(f"{prefix}_flat_shape", np.array([0, 0, 0, dim], dtype=np.int64))])
    return builder.node("Add", [rotated, builder.node("Mul", [rotated_half, sin])])


def _attention(
    builder: GraphBuilder,
    value: str,
    module: Any,
    sequence_length: int,
    prefix: str,
    query_block: int,
    attention_op: bool,
) -> str:
    normalized = _rms_norm(builder, value, module.norm, f"{prefix}_norm")
    qkv = _linear(builder, normalized, module.to_qkv, f"{prefix}_qkv")
    qkv = builder.node(
        "Reshape",
        [qkv, builder.initializer(f"{prefix}_qkv_shape", np.array([0, sequence_length, 3, 8, 64], dtype=np.int64))],
    )
    qkv = builder.node("Transpose", [qkv], perm=[0, 2, 3, 1, 4])
    split = builder.node("Split", [qkv, builder.initializer(f"{prefix}_split_sizes", np.array([1, 1, 1], dtype=np.int64))], axis=1, outputs=[builder.name(f"{prefix}_q"), builder.name(f"{prefix}_k"), builder.name(f"{prefix}_v")])
    q, k, v = [builder.node("Squeeze", [item, builder.initializer(f"{prefix}_squeeze_axes_{index}", np.array([1], dtype=np.int64))]) for index, item in enumerate(split)]
    q = _rotary(builder, q, sequence_length, 64, f"{prefix}_q_rope")
    k = _rotary(builder, k, sequence_length, 64, f"{prefix}_k_rope")
    if attention_op:
        # ONNX's standard Attention operator lets ORT/TensorRT use a fused,
        # memory-efficient scaled-dot-product kernel instead of materializing
        # the full score matrix.
        attended = builder.node(
            "Attention",
            [q, k, v],
            scale=64.0 ** -0.5,
            is_causal=0,
            q_num_heads=8,
            kv_num_heads=8,
            softmax_precision=builder.tensor_proto.FLOAT,
            prefix=f"{prefix}_attention",
        )
    else:
        keys = builder.node("Transpose", [k], perm=[0, 1, 3, 2])
        # Exact fallback for runtimes without the standard Attention op.  A
        # full [bands, heads, T, T] score tensor is ~8.5 GB for the reference
        # 1,722-frame time attention, so split only the query axis.
        block_size = max(1, min(int(query_block), sequence_length))
        block_sizes = [
            min(block_size, sequence_length - start)
            for start in range(0, sequence_length, block_size)
        ]
        query_blocks = builder.node(
            "Split",
            [q, builder.initializer(f"{prefix}_query_block_sizes", np.asarray(block_sizes, dtype=np.int64))],
            axis=2,
            outputs=[builder.name(f"{prefix}_query_block_{index}") for index in range(len(block_sizes))],
        )
        attended_blocks = []
        for index, query_block_value in enumerate(query_blocks):
            scores = builder.node("MatMul", [query_block_value, keys], prefix=f"{prefix}_block{index}_scores")
            scores = builder.node(
                "Mul",
                [scores, builder.constant(np.asarray(64.0 ** -0.5, dtype=np.float32), f"{prefix}_block{index}_attention_scale")],
            )
            scores = builder.node("Softmax", [scores], axis=-1, prefix=f"{prefix}_block{index}_softmax")
            attended_blocks.append(builder.node("MatMul", [scores, v], prefix=f"{prefix}_block{index}_attended"))
        attended = attended_blocks[0] if len(attended_blocks) == 1 else builder.node("Concat", attended_blocks, axis=2, prefix=f"{prefix}_attended_concat")
    gates = _linear(builder, normalized, module.to_gates, f"{prefix}_gates")
    gates = builder.node("Sigmoid", [gates])
    gates = builder.node("Transpose", [gates], perm=[0, 2, 1])
    gates = builder.node("Unsqueeze", [gates, builder.initializer(f"{prefix}_gate_axes", np.array([-1], dtype=np.int64))])
    attended = builder.node("Mul", [attended, gates])
    attended = builder.node("Transpose", [attended], perm=[0, 2, 1, 3])
    attended = builder.node("Reshape", [attended, builder.initializer(f"{prefix}_out_shape", np.array([0, sequence_length, 512], dtype=np.int64))])
    return _linear(builder, attended, module.to_out[0], f"{prefix}_out")


def _gelu(builder: GraphBuilder, value: str, prefix: str) -> str:
    half = builder.constant(np.asarray(0.5, dtype=np.float32), f"{prefix}_half")
    one = builder.constant(np.asarray(1.0, dtype=np.float32), f"{prefix}_one")
    inv_sqrt_two = builder.constant(np.asarray(2.0 ** -0.5, dtype=np.float32), f"{prefix}_inv_sqrt_two")
    erf = builder.node("Erf", [builder.node("Mul", [value, inv_sqrt_two])])
    return builder.node("Mul", [builder.node("Mul", [value, half]), builder.node("Add", [one, erf])])


def _feed_forward(builder: GraphBuilder, value: str, module: Any, prefix: str) -> str:
    normalized = _rms_norm(builder, value, module.net[0], f"{prefix}_norm")
    first = _linear(builder, normalized, module.net[1], f"{prefix}_first")
    activated = _gelu(builder, first, f"{prefix}_gelu")
    # FeedForward is RMSNorm, Linear, GELU, Dropout, Linear, Dropout.
    second = _linear(builder, activated, module.net[4], f"{prefix}_second")
    return second


def _transformer(
    builder: GraphBuilder,
    value: str,
    transformer: Any,
    sequence_length: int,
    prefix: str,
    query_block: int,
    attention_op: bool,
) -> str:
    for index, (attention, feed_forward) in enumerate(transformer.layers):
        attention_output = _attention(builder, value, attention, sequence_length, f"{prefix}_attn{index}", query_block, attention_op)
        value = builder.node("Add", [value, attention_output])
        value = builder.node("Add", [value, _feed_forward(builder, value, feed_forward, f"{prefix}_ff{index}")])
    if not isinstance(transformer.norm, torch.nn.Identity):
        value = _rms_norm(builder, value, transformer.norm, f"{prefix}_final_norm")
    return value


def _band_split(builder: GraphBuilder, value: str, module: Any) -> str:
    pieces = builder.node("Split", [value, builder.initializer("band_split_sizes", np.array(list(module.dim_inputs), dtype=np.int64))], axis=-1, outputs=[builder.name(f"band_input_{index}") for index in range(len(module.dim_inputs))])
    features = []
    for index, (piece, network) in enumerate(zip(pieces, module.to_features)):
        normalized = _rms_norm(builder, piece, network[0], f"band{index}_norm")
        features.append(builder.node("Unsqueeze", [_linear(builder, normalized, network[1], f"band{index}_linear"), builder.initializer(f"band{index}_axes", np.array([2], dtype=np.int64))]))
    return builder.node("Concat", features, axis=2)


def _mask_head(builder: GraphBuilder, value: str, head: Any, prefix: str) -> str:
    values = []
    for index, network in enumerate(head.to_freqs):
        band = builder.node("Gather", [value, builder.constant(np.asarray(index, dtype=np.int64), f"{prefix}_band{index}")], axis=2)
        first = _linear(builder, band, network[0][0], f"{prefix}_band{index}_first")
        hidden = builder.node("Tanh", [first])
        projection = _linear(builder, hidden, network[0][2], f"{prefix}_band{index}_projection")
        split = builder.node("Split", [projection, builder.initializer(f"{prefix}_band{index}_split_sizes", np.array([network[0][2].out_features // 2] * 2, dtype=np.int64))], axis=-1, outputs=[builder.name(f"{prefix}_band{index}_a"), builder.name(f"{prefix}_band{index}_b")])
        values.append(builder.node("Mul", [split[0], builder.node("Sigmoid", [split[1]])]))
    return builder.node("Concat", values, axis=-1)


def build_model(
    model: torch.nn.Module,
    config: SeparationConfig,
    batch: int,
    frames: int,
    output_path: Path,
    *,
    opset: int = 24,
    attention_mode: str = "attention",
    precision: str = "fp32",
    input_precision: str | None = None,
) -> dict[str, object]:
    import onnx
    from onnx import TensorProto, helper

    core = SpectralCore(model).eval()
    set_math_attention(core)
    if attention_mode not in {"attention", "blocked"}:
        raise ValueError(f"unsupported attention mode: {attention_mode}")
    attention_op = attention_mode == "attention"
    if attention_op and opset < 23:
        raise ValueError("the standard Attention operator requires ONNX opset 23 or newer")
    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"unsupported precision: {precision}")
    if input_precision is None:
        input_precision = "fp32" if precision == "fp16" else "fp32"
    if input_precision not in {"fp32", "fp16"}:
        raise ValueError(f"unsupported input precision: {input_precision}")
    float_dtype = np.dtype(np.float16 if precision == "fp16" else np.float32)
    tensor_type = TensorProto.FLOAT16 if precision == "fp16" else TensorProto.FLOAT
    input_tensor_type = TensorProto.FLOAT16 if input_precision == "fp16" else TensorProto.FLOAT
    builder = GraphBuilder(helper, TensorProto, [], [], float_dtype)
    input_name = "stft"
    input_value = builder.name("input")
    builder.nodes.append(helper.make_node("Identity", [input_name], [input_value]))
    value = builder.node("Transpose", [input_value], perm=[0, 2, 1, 3])
    value = builder.node("Reshape", [value, builder.initializer("input_flat_shape", np.array([0, frames, config.spectral_channels * 2], dtype=np.int64))])
    value = _band_split(builder, value, core.band_split)

    for layer_index, transformer_block in enumerate(core.layers):
        if len(transformer_block) != 2:
            raise NotImplementedError("linear transformer blocks are not supported by the direct builder")
        time_transformer, frequency_transformer = transformer_block
        bands = len(core.band_split.to_features)
        value = builder.node("Transpose", [value], perm=[0, 2, 1, 3])
        value = builder.node("Reshape", [value, builder.initializer(f"time_input_shape_{layer_index}", np.array([batch * bands, frames, 256], dtype=np.int64))])
        value = _transformer(builder, value, time_transformer, frames, f"layer{layer_index}_time", config.attention_query_block, attention_op)
        value = builder.node("Reshape", [value, builder.initializer(f"time_output_shape_{layer_index}", np.array([batch, bands, frames, 256], dtype=np.int64))])
        value = builder.node("Transpose", [value], perm=[0, 2, 1, 3])
        value = builder.node("Reshape", [value, builder.initializer(f"frequency_input_shape_{layer_index}", np.array([batch * frames, bands, 256], dtype=np.int64))])
        value = _transformer(builder, value, frequency_transformer, bands, f"layer{layer_index}_frequency", config.attention_query_block, attention_op)
        value = builder.node("Reshape", [value, builder.initializer(f"frequency_output_shape_{layer_index}", np.array([batch, frames, bands, 256], dtype=np.int64))])

    value = _rms_norm(builder, value, core.final_norm, "final")
    heads = [_mask_head(builder, value, head, f"mask{index}") for index, head in enumerate(core.mask_estimators)]
    heads = [
        builder.node(
            "Reshape",
            [head, builder.initializer(f"mask{index}_shape", np.array([batch, frames, config.spectral_channels, 2], dtype=np.int64))],
        )
        for index, head in enumerate(heads)
    ]
    heads = [builder.node("Unsqueeze", [head, builder.initializer(f"mask{index}_axes", np.array([1], dtype=np.int64))]) for index, head in enumerate(heads)]
    output_name = "mask"
    builder.nodes.append(helper.make_node("Concat", heads, [output_name], axis=1))

    graph = helper.make_graph(
        builder.nodes,
        "bs_roformer_spectral_core",
        [helper.make_tensor_value_info(input_name, input_tensor_type, [batch, config.spectral_channels, frames, 2])],
        [helper.make_tensor_value_info(output_name, tensor_type, [batch, config.num_stems, frames, config.spectral_channels, 2])],
        initializer=builder.initializers,
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], producer_name="vocalarc-onnx")
    onnx.checker.check_model(onnx_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onnx_model, str(output_path), save_as_external_data=False)
    return {
        "input": [batch, config.spectral_channels, frames, 2],
        "output": [batch, config.num_stems, frames, config.spectral_channels, 2],
        "nodes": len(builder.nodes),
        "initializers": len(builder.initializers),
        "attentionMode": attention_mode,
        "attentionQueryBlock": config.attention_query_block if attention_mode == "blocked" else None,
        "precision": precision,
        "inputPrecision": input_precision,
    }
