from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch

from knnvc.backend import KNNVCBackend
from knnvc.config import AppConfig, AudioConfig, ModelConfig


class HubModel:
    wavlm: HubModel

    def __init__(self) -> None:
        self.wavlm = self

    def extract_features(
        self,
        waveform: torch.Tensor,
        *,
        output_layer: int,
        ret_layer_results: bool,
    ) -> tuple[torch.Tensor, None]:
        assert output_layer == 6
        assert ret_layer_results is False
        return waveform[:, :8].reshape(1, 2, 4), None

    def vocode(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=-1).repeat_interleave(320, dim=1)


def test_backend_passes_device_and_uses_released_model(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    def fake_load(repository: str, entrypoint: str, **kwargs: Any) -> HubModel:
        calls.update(repository=repository, entrypoint=entrypoint, **kwargs)
        return HubModel()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    config = AppConfig(
        model=ModelConfig(repository="example/model:tag", device="cpu"),
        audio=AudioConfig(output_loudness_db=None),
    )
    backend = KNNVCBackend(config)
    monkeypatch.setattr(backend, "_verify_checkpoints", lambda: None)

    features = backend.extract_features(torch.arange(8, dtype=torch.float32).unsqueeze(0))
    waveform = backend.synthesize(features)

    assert features.shape == (2, 4)
    assert waveform.shape == (1, 640)
    assert calls["repository"] == "example/model:tag"
    assert calls["entrypoint"] == "knn_vc"
    assert calls["device"] == "cpu"
    assert calls["pretrained"] is True
    assert calls["prematched"] is True


def test_backend_rejects_invalid_features(monkeypatch: Any) -> None:
    class EmptyModel(HubModel):
        def extract_features(
            self,
            waveform: torch.Tensor,
            *,
            output_layer: int,
            ret_layer_results: bool,
        ) -> tuple[torch.Tensor, None]:
            del waveform, output_layer, ret_layer_results
            return torch.empty(1, 0, 4), None

    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: EmptyModel())
    backend = KNNVCBackend(
        AppConfig(model=ModelConfig(device="cpu"), audio=AudioConfig(output_loudness_db=None))
    )
    monkeypatch.setattr(backend, "_verify_checkpoints", lambda: None)
    try:
        backend.extract_features(torch.ones(1, 640))
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("Expected invalid WavLM output to be rejected")


def test_backend_verifies_checkpoint_fingerprints(tmp_path: Path, monkeypatch: Any) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    wavlm = checkpoint_dir / "WavLM-Large.pt"
    vocoder = checkpoint_dir / "prematch_g_02500000.pt"
    wavlm.write_bytes(b"wavlm")
    vocoder.write_bytes(b"vocoder")
    backend = KNNVCBackend(AppConfig(model=ModelConfig(device="cpu")))
    backend.wavlm_sha256 = hashlib.sha256(b"wavlm").hexdigest()
    backend.vocoder_sha256 = hashlib.sha256(b"vocoder").hexdigest()
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))

    backend._verify_checkpoints()
    vocoder.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="integrity"):
        backend._verify_checkpoints()
