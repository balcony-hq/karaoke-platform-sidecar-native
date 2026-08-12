"""Torch audio front-end and upstream-compatible chunk aggregation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from .config import SeparationConfig
except ImportError:  # direct script imports from the onnx directory
    from config import SeparationConfig  # type: ignore[no-redef]


def _window(config: SeparationConfig, device: torch.device) -> Tensor:
    return torch.hann_window(config.win_length, device=device, dtype=torch.float32)


def waveform_to_stft(waveform: Tensor, config: SeparationConfig) -> Tensor:
    """Return the upstream layout ``[B, F*C, T, 2]``."""

    if waveform.ndim != 3 or waveform.shape[1] != config.channels:
        raise ValueError(f"expected [batch, {config.channels}, samples], got {tuple(waveform.shape)}")
    batch, channels, length = waveform.shape
    flat = waveform.reshape(batch * channels, length)
    stft = torch.stft(
        flat,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=_window(config, waveform.device),
        normalized=False,
        return_complex=True,
    )
    stft = torch.view_as_real(stft).reshape(batch, channels, config.frequency_bins, -1, 2)
    return stft.permute(0, 2, 1, 3, 4).reshape(batch, config.spectral_channels, -1, 2)


def stft_to_waveform(
    stft: Tensor,
    mask: Tensor,
    length: int,
    config: SeparationConfig,
) -> Tensor:
    """Apply a real/imag mask and return ``[B, stems, channels, samples]``."""

    if stft.ndim != 4 or mask.ndim != 5:
        raise ValueError("expected STFT [B,F*C,T,2] and mask [B,N,T,F*C,2]")
    batch, frequency_channels, frames, components = stft.shape
    if mask.shape[0] != batch or mask.shape[2] != frames or mask.shape[3] != frequency_channels or mask.shape[4] != components:
        raise ValueError(f"STFT and mask shapes differ: {tuple(stft.shape)} vs {tuple(mask.shape)}")
    stems = mask.shape[1]
    mask = mask.permute(0, 1, 3, 2, 4)
    stft_complex = torch.view_as_complex(stft)
    mask_complex = torch.view_as_complex(mask)
    masked = stft_complex.unsqueeze(1) * mask_complex
    masked = masked.reshape(batch, stems, config.frequency_bins, config.channels, frames)
    masked = masked.permute(0, 1, 3, 2, 4).reshape(
        batch * stems * config.channels, config.frequency_bins, frames
    )
    if config.zero_dc:
        masked[:, 0, :] = 0
    audio = torch.istft(
        masked,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=_window(config, masked.device),
        normalized=False,
        length=length,
    )
    return audio.reshape(batch, stems, config.channels, length)


def predict_with_core(
    core: Any,
    waveform: Tensor,
    config: SeparationConfig,
) -> Tensor:
    """Run a Torch module or ONNX session on a fixed-size waveform batch."""

    if isinstance(core, torch.nn.Module):
        stft = waveform_to_stft(waveform, config)
        parameter = next(core.parameters(), None)
        model_stft = stft.to(dtype=parameter.dtype) if parameter is not None else stft
        mask = core(model_stft).float()
    else:
        # ONNX Runtime returns NumPy output even for CUDA providers.  Keep
        # the waveform transform on the same accelerator as the reference
        # model when possible; CPU and CUDA ISTFT use different numerical
        # paths, which is especially visible for very low-energy signals.
        providers = set(core.get_providers()) if hasattr(core, "get_providers") else set()
        accelerated = providers.intersection({"CUDAExecutionProvider", "TensorrtExecutionProvider"})
        audio_device = torch.device("cuda") if accelerated and torch.cuda.is_available() else waveform.device
        model_waveform = waveform.to(audio_device)
        stft = waveform_to_stft(model_waveform, config)
        input_name = core.get_inputs()[0].name
        output_name = core.get_outputs()[0].name
        input_type = str(getattr(core.get_inputs()[0], "type", ""))
        input_dtype = np.float16 if "float16" in input_type else np.float32
        input_values = stft.detach().cpu().numpy().astype(input_dtype, copy=False)
        declared_batch = core.get_inputs()[0].shape[0]
        # Static provider graphs are intentionally exported for batch one or
        # two.  The overlap scheduler can end with a short batch, so split a
        # mismatched request into valid provider calls rather than padding it
        # and accidentally changing the reference arithmetic.
        if isinstance(declared_batch, int) and declared_batch != stft.shape[0]:
            masks = []
            for item in input_values:
                masks.append(
                    core.run([output_name], {input_name: item[None, ...]})[0][0]
                )
            mask = torch.from_numpy(np.stack(masks, axis=0)).to(audio_device).float()
        else:
            mask = torch.from_numpy(core.run([output_name], {input_name: input_values})[0]).to(audio_device).float()
    return stft_to_waveform(stft, mask, waveform.shape[-1], config)[:, 0].cpu()


def _pad_part(part: Tensor, target: int) -> Tensor:
    amount = target - part.shape[-1]
    if amount < 0:
        raise ValueError("part is longer than the configured chunk")
    if amount == 0:
        return part
    pad_mode = "reflect" if part.shape[-1] > target // 2 else "constant"
    return F.pad(part, (0, amount), mode=pad_mode, value=0.0)


def demix(
    predict: Callable[[Tensor], Tensor],
    mix: np.ndarray,
    config: SeparationConfig,
    *,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> np.ndarray:
    """Port the upstream generic ``demix`` overlap/fade algorithm."""

    if mix.ndim != 2 or mix.shape[0] != config.channels:
        raise ValueError(f"expected [{config.channels}, samples], got {mix.shape}")
    chunk_size = config.chunk_size
    step = config.step_size
    fade_size = chunk_size // 10
    border = chunk_size - step
    length_init = mix.shape[-1]
    working = torch.from_numpy(mix.astype(np.float32, copy=False))
    if length_init > 2 * border and border > 0:
        working = F.pad(working, (border, border), mode="reflect")

    window = torch.ones(chunk_size, dtype=torch.float32)
    window[-fade_size:] = torch.linspace(1.0, 0.0, fade_size)
    result = torch.zeros_like(working)
    counter = torch.zeros_like(working)
    locations: list[tuple[int, int]] = []
    parts: list[Tensor] = []
    position = 0
    while position < working.shape[-1]:
        part = working[:, position : position + chunk_size]
        length = part.shape[-1]
        parts.append(_pad_part(part, chunk_size))
        locations.append((position, length))
        position += step
        if len(parts) >= config.inference_batch_size or position >= working.shape[-1]:
            batch = torch.stack(parts)
            estimates = predict(batch).detach().cpu()
            active_window = window.clone()
            if position - step == 0:
                active_window[:fade_size] = 1.0
            elif position >= working.shape[-1]:
                active_window[-fade_size:] = 1.0
            for estimate, (start, length) in zip(estimates, locations):
                result[:, start : start + length] += estimate[:, :length] * active_window[:length]
                counter[:, start : start + length] += active_window[:length]
            if progress:
                progress({"stage": "chunks", "completedSamples": min(position, int(working.shape[-1]))})
            parts.clear()
            locations.clear()

    output = (result / counter.clamp_min(1e-12)).numpy()
    np.nan_to_num(output, copy=False, nan=0.0)
    if length_init > 2 * border and border > 0:
        output = output[..., border:-border]
    return output


def bigshifts(
    predict: Callable[[Tensor], Tensor],
    mix: np.ndarray,
    config: SeparationConfig,
    count: int,
    *,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> np.ndarray:
    """Match upstream circular-shift inference and averaging."""

    count = max(1, int(count))
    shift_size = mix.shape[1] // count
    results: list[np.ndarray] = []
    for index in range(count):
        shift = index * shift_size
        shifted = np.concatenate((mix[:, -shift:], mix[:, :-shift]), axis=-1)
        result = demix(predict, shifted, config, progress=progress)
        unshifted = np.concatenate((result[..., shift:], result[..., :shift]), axis=-1)
        results.append(unshifted)
        if progress:
            progress({"stage": "bigshifts", "completed": index + 1, "total": count})
    return np.mean(results, axis=0)


def apply_tta(
    predict: Callable[[Tensor], Tensor],
    mix: np.ndarray,
    base: np.ndarray,
    config: SeparationConfig,
    bigshift_count: int,
    *,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> np.ndarray:
    """Match upstream channel-swap and polarity-inversion TTA."""

    augmented = [mix[::-1].copy(), -1.0 * mix.copy()]
    result = base.copy()
    for index, variant in enumerate(augmented):
        estimate = bigshifts(predict, variant, config, bigshift_count, progress=progress)
        if index == 0:
            result += estimate[::-1].copy()
        else:
            result -= estimate
    return result / (len(augmented) + 1)
