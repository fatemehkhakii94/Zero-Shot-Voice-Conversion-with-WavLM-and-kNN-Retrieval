from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchaudio


class FakeBackend:
    repository = "example/knn-vc:test"
    feature_layer = 6
    device = torch.device("cpu")
    wavlm_sha256 = "a" * 64
    vocoder_sha256 = "b" * 64

    def extract_features(
        self,
        waveform: torch.Tensor,
        *,
        vad_trigger_db: float = 0.0,
    ) -> torch.Tensor:
        del vad_trigger_db
        samples = waveform.squeeze(0)
        usable = max(4, samples.numel() - samples.numel() % 4)
        return samples[:usable].reshape(-1, 4).to(torch.float32)

    def synthesize(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=1).repeat_interleave(320).unsqueeze(0)


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


def write_test_audio(path: Path, *, sample_rate: int = 16_000, seconds: float = 0.2) -> Path:
    samples = int(sample_rate * seconds)
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    waveform = (0.1 * torch.sin(2 * torch.pi * 220 * time)).unsqueeze(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform, sample_rate)
    return path
