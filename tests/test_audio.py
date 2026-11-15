from pathlib import Path

import pytest
import torch
import torchaudio

from knnvc.audio import load_audio, save_audio, stabilize_waveform


def test_load_audio_resamples_and_downmixes(tmp_path: Path) -> None:
    waveform = torch.stack((torch.ones(800), torch.zeros(800)))
    path = tmp_path / "stereo.wav"
    torchaudio.save(str(path), waveform, 8_000)

    loaded = load_audio(path, sample_rate=16_000)

    assert loaded.shape[0] == 1
    assert loaded.shape[1] == 1_600
    assert loaded.dtype == torch.float32


def test_stabilize_waveform_limits_peak() -> None:
    stabilized = stabilize_waveform(torch.tensor([[2.0, -1.0]]))
    assert stabilized.abs().max().item() == pytest.approx(0.999)


def test_save_audio_refuses_implicit_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "output.wav"
    waveform = torch.zeros(1, 1_000)
    save_audio(path, waveform)
    with pytest.raises(FileExistsError):
        save_audio(path, waveform)
