from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import torch
import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class ModelConfig:
    repository: str = "bshall/knn-vc:v0.1"
    feature_layer: int = 6
    device: str = "auto"


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    source_max_seconds: float | None = None
    reference_max_seconds: float | None = None
    reference_vad_trigger_db: float = 7.0
    output_loudness_db: float | None = -16.0


@dataclass(frozen=True)
class MatchingConfig:
    k: int = 4
    metric: str = "cosine"
    device: str = "auto"
    query_chunk_size: int = 512
    reference_chunk_size: int = 32_768


@dataclass(frozen=True)
class PathConfig:
    bank_dir: str = "artifacts/banks"
    audio_dir: str = "artifacts/audio"
    report_dir: str = "artifacts/reports"


@dataclass(frozen=True)
class EvaluationConfig:
    whisper_model: str = "base.en"
    language: str = "en"
    timing_warmup_runs: int = 1
    timing_runs: int = 3


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = ModelConfig()
    audio: AudioConfig = AudioConfig()
    matching: MatchingConfig = MatchingConfig()
    paths: PathConfig = PathConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    def validate(self) -> None:
        errors: list[str] = []
        if self.model.feature_layer != 6:
            errors.append("model.feature_layer must be 6 for the released prematched HiFi-GAN")
        if self.audio.sample_rate != 16_000:
            errors.append("audio.sample_rate must be 16000 for WavLM and the released vocoder")
        if self.matching.k < 1:
            errors.append("matching.k must be at least 1")
        if self.matching.metric not in {"cosine", "l2"}:
            errors.append("matching.metric must be 'cosine' or 'l2'")
        if self.matching.query_chunk_size < 1 or self.matching.reference_chunk_size < 1:
            errors.append("matching chunk sizes must be positive")
        if self.evaluation.timing_warmup_runs < 0:
            errors.append("evaluation.timing_warmup_runs cannot be negative")
        if self.evaluation.timing_runs < 1:
            errors.append("evaluation.timing_runs must be at least 1")
        for name, value in (
            ("audio.source_max_seconds", self.audio.source_max_seconds),
            ("audio.reference_max_seconds", self.audio.reference_max_seconds),
        ):
            if value is not None and value <= 0:
                errors.append(f"{name} must be positive or null")
        resolve_device(self.model.device)
        resolve_device(self.matching.device)
        if errors:
            raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but CUDA is not available: {value}")
    return device


def _build_dataclass(cls: type[T], values: dict[str, Any], section: str) -> T:
    allowed = {field.name for field in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(values) - allowed)
    if unknown:
        joined = ", ".join(f"{section}.{key}" for key in unknown)
        raise ValueError(f"Unknown configuration key(s): {joined}")
    return cls(**values)


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        config = AppConfig()
        config.validate()
        return config

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("The configuration root must be a mapping")

    section_types = get_type_hints(AppConfig)
    unknown_sections = sorted(set(raw) - set(section_types))
    if unknown_sections:
        raise ValueError(f"Unknown configuration section(s): {', '.join(unknown_sections)}")

    sections: dict[str, Any] = {}
    for section, section_type in section_types.items():
        values = raw.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping")
        sections[section] = _build_dataclass(section_type, values, section)

    config = AppConfig(**sections)
    config.validate()
    return config
