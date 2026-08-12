"""Stable inference settings shared by export, benchmark, and sidecar code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeparationConfig:
    """The reference settings from ``leap_xe_config_voc.yaml``.

    The model is intentionally configured here instead of re-reading the
    training YAML in the runtime.  The exported model manifest records these
    values and prevents a model from silently being used with incompatible
    DSP settings.
    """

    sample_rate: int = 44_100
    channels: int = 2
    n_fft: int = 2_048
    hop_length: int = 512
    win_length: int = 2_048
    chunk_size: int = 881_559
    num_overlap: int = 2
    inference_batch_size: int = 2
    num_stems: int = 1
    zero_dc: bool = True
    attention_query_block: int = 128
    default_bigshifts: int = 2
    default_tta: bool = True

    @property
    def frequency_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def spectral_channels(self) -> int:
        return self.frequency_bins * self.channels

    @property
    def chunk_frames(self) -> int:
        # torch.stft(center=True) uses this exact frame count for the
        # reference chunk size.
        return self.chunk_size // self.hop_length + 1

    @property
    def step_size(self) -> int:
        return self.chunk_size // self.num_overlap

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "chunk_size": self.chunk_size,
            "num_overlap": self.num_overlap,
            "inference_batch_size": self.inference_batch_size,
            "num_stems": self.num_stems,
            "zero_dc": self.zero_dc,
            "attention_query_block": self.attention_query_block,
            "default_bigshifts": self.default_bigshifts,
            "default_tta": self.default_tta,
            "frequency_bins": self.frequency_bins,
            "spectral_channels": self.spectral_channels,
            "chunk_frames": self.chunk_frames,
            "step_size": self.step_size,
        }
