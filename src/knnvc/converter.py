from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from knnvc.audio import load_audio, save_audio
from knnvc.backend import KNNVCBackend, SpeechBackend
from knnvc.bank import BankStore, validate_bank_compatibility
from knnvc.config import AppConfig
from knnvc.matching import KNNMatcher


@dataclass(frozen=True)
class ConversionResult:
    source: str
    target_bank: str
    output: str
    source_duration_seconds: float
    output_duration_seconds: float
    source_frames: int
    reference_frames: int
    feature_dimension: int
    neighbors: int
    metric: str
    inference_seconds: float
    wall_seconds: float
    real_time_factor: float
    model_repository: str
    wavlm_sha256: str
    vocoder_sha256: str
    device: str
    torch_version: str
    python_version: str

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


class VoiceConverter:
    def __init__(
        self,
        config: AppConfig,
        *,
        backend: SpeechBackend | None = None,
        bank_store: BankStore | None = None,
    ):
        config.validate()
        self.config = config
        self.backend = backend or KNNVCBackend(config)
        self.bank_store = bank_store or BankStore(config.paths.bank_dir)
        self.matcher = KNNMatcher(config.matching)

    def convert(
        self,
        source_path: str | Path,
        target_bank: str,
        output_path: str | Path | None = None,
        *,
        overwrite: bool = False,
        write_metadata: bool = True,
    ) -> ConversionResult:
        source = Path(source_path)
        output = Path(output_path) if output_path else self._default_output(source, target_bank)
        metadata_path = output.with_suffix(".json")
        if not overwrite:
            existing = [path for path in (output, metadata_path) if path.exists()]
            if existing:
                raise FileExistsError(f"Output already exists: {existing[0]}")

        wall_start = time.perf_counter()
        waveform = load_audio(
            source,
            sample_rate=self.config.audio.sample_rate,
            max_seconds=self.config.audio.source_max_seconds,
        )
        bank = self.bank_store.load(target_bank)
        validate_bank_compatibility(bank, self.backend, self.config)

        inference_start = time.perf_counter()
        source_features = self.backend.extract_features(waveform)
        matched_features, match_summary = self.matcher.match(source_features, bank.features)
        converted = self.backend.synthesize(matched_features)
        inference_seconds = time.perf_counter() - inference_start

        save_audio(
            output,
            converted,
            sample_rate=self.config.audio.sample_rate,
            overwrite=overwrite,
        )
        wall_seconds = time.perf_counter() - wall_start
        source_duration = waveform.shape[1] / self.config.audio.sample_rate
        output_duration = converted.shape[1] / self.config.audio.sample_rate
        result = ConversionResult(
            source=str(source),
            target_bank=target_bank,
            output=str(output),
            source_duration_seconds=round(source_duration, 6),
            output_duration_seconds=round(output_duration, 6),
            source_frames=match_summary.query_frames,
            reference_frames=match_summary.reference_frames,
            feature_dimension=match_summary.feature_dimension,
            neighbors=match_summary.k,
            metric=match_summary.metric,
            inference_seconds=round(inference_seconds, 6),
            wall_seconds=round(wall_seconds, 6),
            real_time_factor=round(inference_seconds / source_duration, 6),
            model_repository=self.backend.repository,
            wavlm_sha256=self.backend.wavlm_sha256,
            vocoder_sha256=self.backend.vocoder_sha256,
            device=str(self.backend.device),
            torch_version=torch.__version__,
            python_version=platform.python_version(),
        )
        if write_metadata:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(result.to_dict(), handle, indent=2)
                handle.write("\n")
        return result

    def _default_output(self, source: Path, target_bank: str) -> Path:
        return Path(self.config.paths.audio_dir) / f"{source.stem}--to--{target_bank}.wav"
