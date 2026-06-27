import torch
import torchaudio
from pathlib import Path


def load_audio(path: str | Path, sample_rate: int = 16000, max_duration_sec: float = 10.0) -> torch.Tensor:
    """Load audio file, resample to target sample_rate, convert to mono, clip to max_duration_sec.
    Returns: (1, T) float32 tensor normalised to [-1, 1].
    """
    wav, sr = torchaudio.load(str(path))

    # Resample if needed
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    # Mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Clip
    max_samples = int(max_duration_sec * sample_rate)
    wav = wav[:, :max_samples]

    return wav  # (1, T)
