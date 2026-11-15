from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol, cast

import torch
import torchaudio

from knnvc.audio import trim_silence_vad
from knnvc.config import AppConfig, resolve_device


class SpeechBackend(Protocol):
    repository: str
    feature_layer: int
    device: torch.device
    wavlm_sha256: str
    vocoder_sha256: str

    def extract_features(
        self, waveform: torch.Tensor, *, vad_trigger_db: float = 0.0
    ) -> torch.Tensor: ...

    def synthesize(self, features: torch.Tensor) -> torch.Tensor: ...


class KNNVCBackend:
    """Pinned adapter around the released kNN-VC WavLM and HiFi-GAN models."""

    wavlm_sha256 = "6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f"
    vocoder_sha256 = "f924c7632c6eaf99004386d62293e124419e33582573552a6ae976eb88ed2dd5"

    def __init__(self, config: AppConfig):
        self.repository = config.model.repository
        self.feature_layer = config.model.feature_layer
        self.device = resolve_device(config.model.device)
        self.sample_rate = config.audio.sample_rate
        self.output_loudness_db = config.audio.output_loudness_db
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            self._model = torch.hub.load(  # type: ignore[no-untyped-call]
                self.repository,
                "knn_vc",
                pretrained=True,
                prematched=True,
                trust_repo=True,
                device=str(self.device),
            )
            self._verify_checkpoints()
        return self._model

    def _verify_checkpoints(self) -> None:
        checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
        expected = {
            "WavLM-Large.pt": self.wavlm_sha256,
            "prematch_g_02500000.pt": self.vocoder_sha256,
        }
        for filename, expected_hash in expected.items():
            path = checkpoint_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Downloaded model checkpoint is missing: {path}")
            actual_hash = _sha256_file(path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Checkpoint integrity verification failed for {filename}: "
                    f"expected {expected_hash}, received {actual_hash}"
                )

    @torch.inference_mode()
    def extract_features(
        self,
        waveform: torch.Tensor,
        *,
        vad_trigger_db: float = 0.0,
    ) -> torch.Tensor:
        model = self._load()
        waveform = trim_silence_vad(
            waveform,
            sample_rate=self.sample_rate,
            trigger_db=vad_trigger_db,
        )
        extracted = model.wavlm.extract_features(
            waveform.to(self.device),
            output_layer=self.feature_layer,
            ret_layer_results=False,
        )
        features = cast(torch.Tensor, extracted[0]).squeeze(0)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("WavLM returned an empty or invalid feature sequence")
        return features.detach().to(device="cpu", dtype=torch.float32).contiguous()

    @torch.inference_mode()
    def synthesize(self, features: torch.Tensor) -> torch.Tensor:
        model = self._load()
        waveform = cast(
            torch.Tensor,
            model.vocode(features.unsqueeze(0).to(self.device)),
        ).cpu()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError(f"HiFi-GAN returned an unexpected shape: {tuple(waveform.shape)}")
        if self.output_loudness_db is not None and waveform.shape[1] >= self.sample_rate // 10:
            loudness = torchaudio.functional.loudness(waveform, self.sample_rate)
            if torch.isfinite(loudness).all():
                waveform = cast(
                    torch.Tensor,
                    torchaudio.functional.gain(
                        waveform,
                        self.output_loudness_db - loudness,
                    ),
                )
        return waveform.to(torch.float32).contiguous()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
