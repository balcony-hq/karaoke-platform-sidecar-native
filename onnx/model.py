"""Reference model loading and the provider-neutral spectral ONNX graph."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2] / "Music-Source-Separation-Training"


def _add_source_root() -> Path:
    root = _source_root()
    if not root.is_dir():
        raise FileNotFoundError(f"upstream source checkout is missing: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _state_dict(checkpoint: Any) -> dict[str, Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "state"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(key, str) for key in checkpoint):
            return checkpoint
    raise ValueError("checkpoint does not contain a PyTorch state dict")


def load_reference_model(
    config_path: Path,
    checkpoint_path: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, Any]:
    """Load the exact upstream model used by the reference implementation."""

    _add_source_root()
    from utils.settings import get_model_from_config  # type: ignore[import-not-found]

    model, config = get_model_from_config("bs_roformer", str(config_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(_state_dict(checkpoint), strict=True)
    model = model.to(device).eval()
    return model, config


def set_math_attention(module: nn.Module) -> None:
    """Disable PyTorch's fused SDPA branch for a portable ONNX trace.

    The upstream model's math is unchanged; this only selects the explicit
    matmul/softmax implementation while exporting.  It avoids exporting a
    backend-specific fused attention operator and is also useful for a clean
    ONNX-vs-PyTorch comparison.
    """

    for child in module.modules():
        if hasattr(child, "flash"):
            child.flash = False


class SpectralCore(nn.Module):
    """BS-RoFormer from STFT bins to real/imaginary mask values.

    Input shape is ``[batch, frequency*channels, frames, real_imag]`` and
    output shape is ``[batch, stems, frames, frequency*channels, real_imag]``.
    STFT and ISTFT intentionally remain outside ONNX so providers do not need
    complex-number or audio-front-end support.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        required = ("band_split", "layers", "final_norm", "mask_estimators")
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise TypeError(f"model is not the expected BSRoformer: missing {missing}")
        self.band_split = model.band_split
        self.layers = model.layers
        self.final_norm = model.final_norm
        self.mask_estimators = model.mask_estimators
        self.num_stems = int(model.num_stems)

    def forward(self, stft_real: Tensor) -> Tensor:
        batch, frequency_channels, frames, complex_components = stft_real.shape
        if complex_components != 2:
            raise ValueError("the final spectral dimension must contain real and imaginary values")

        x = stft_real.permute(0, 2, 1, 3).reshape(
            batch, frames, frequency_channels * complex_components
        )
        x = self.band_split(x)

        for transformer_block in self.layers:
            if len(transformer_block) == 3:
                linear_transformer, time_transformer, frequency_transformer = transformer_block
                x = x.reshape(batch, frames * x.shape[2], x.shape[3])
                x = linear_transformer(x)
                x = x.reshape(batch, frames, -1, x.shape[-1])
            else:
                time_transformer, frequency_transformer = transformer_block

            # Exact equivalent of the upstream einops packing:
            # b t f d -> b f t d -> (b f) t d.
            bands = x.shape[2]
            dim = x.shape[3]
            x = x.permute(0, 2, 1, 3).reshape(batch * bands, frames, dim)
            x = time_transformer(x)
            x = x.reshape(batch, bands, frames, dim).permute(0, 2, 1, 3)

            # b t f d -> (b t) f d.
            x = x.reshape(batch * frames, bands, dim)
            x = frequency_transformer(x)
            x = x.reshape(batch, frames, bands, dim)

        x = self.final_norm(x)
        mask = torch.stack([head(x) for head in self.mask_estimators], dim=1)
        return mask.reshape(batch, self.num_stems, frames, frequency_channels, 2)

