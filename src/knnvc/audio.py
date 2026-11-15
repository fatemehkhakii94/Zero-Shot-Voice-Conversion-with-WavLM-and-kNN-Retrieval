from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import torchaudio


def load_audio(
    path: str | Path,
    *,
    sample_rate: int = 16_000,
    max_seconds: float | None = None,
) -> torch.Tensor:
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    loaded = torchaudio.load(str(audio_path), normalize=True)
    waveform = cast(torch.Tensor, loaded[0])
    original_rate = int(loaded[1])
    if waveform.ndim != 2 or waveform.shape[1] == 0:
        raise ValueError(f"Audio must contain at least one non-empty channel: {audio_path}")
    if not torch.isfinite(waveform).all():
        raise ValueError(f"Audio contains NaN or infinite samples: {audio_path}")

    waveform = waveform.to(torch.float32)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if original_rate != sample_rate:
        waveform = cast(
            torch.Tensor,
            torchaudio.functional.resample(waveform, original_rate, sample_rate),
        )
    if max_seconds is not None:
        waveform = waveform[:, : int(max_seconds * sample_rate)]
    if waveform.shape[1] < 320:
        raise ValueError(f"Audio is too short to extract WavLM features: {audio_path}")
    return waveform.contiguous()


def stabilize_waveform(waveform: torch.Tensor, peak: float = 0.999) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[0] != 1 or waveform.shape[1] == 0:
        raise ValueError("Output waveform must have shape (1, samples) and be non-empty")
    if not torch.isfinite(waveform).all():
        raise ValueError("Output waveform contains NaN or infinite samples")

    waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
    maximum = waveform.abs().max()
    if maximum > peak:
        waveform = waveform * (peak / maximum)
    return waveform.contiguous()


def trim_silence_vad(
    waveform: torch.Tensor,
    *,
    sample_rate: int = 16_000,
    trigger_db: float = 7.0,
) -> torch.Tensor:
    if trigger_db <= 0:
        return waveform
    vad = torchaudio.transforms.Vad(sample_rate=sample_rate, trigger_level=trigger_db)
    front_trimmed = vad(waveform)
    if front_trimmed.shape[1] == 0:
        raise ValueError("Voice activity detection removed the entire reference recording")
    reversed_audio = torch.flip(front_trimmed, dims=(-1,))
    back_trimmed = torch.flip(vad(reversed_audio), dims=(-1,))
    if back_trimmed.shape[1] == 0:
        raise ValueError("Voice activity detection removed the entire reference recording")
    return back_trimmed.contiguous()


def save_audio(
    path: str | Path,
    waveform: torch.Tensor,
    *,
    sample_rate: int = 16_000,
    overwrite: bool = False,
) -> Path:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists; pass overwrite=True to replace it: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), stabilize_waveform(waveform), sample_rate)
    return output_path
