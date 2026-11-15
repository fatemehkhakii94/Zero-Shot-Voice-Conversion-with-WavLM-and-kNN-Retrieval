from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from knnvc.audio import load_audio
from knnvc.backend import SpeechBackend
from knnvc.config import AppConfig

BANK_FORMAT_VERSION = 1
_BANK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class ReferenceRecord:
    name: str
    sha256: str
    duration_seconds: float


@dataclass(frozen=True)
class BankMetadata:
    format_version: int
    bank_id: str
    created_at: str
    model_repository: str
    wavlm_sha256: str
    vocoder_sha256: str
    feature_layer: int
    sample_rate: int
    vad_trigger_db: float
    reference_count: int
    reference_duration_seconds: float
    feature_frames: int
    feature_dimension: int
    feature_dtype: str
    features_sha256: str
    references: tuple[ReferenceRecord, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BankMetadata:
        values = dict(raw)
        values["references"] = tuple(ReferenceRecord(**item) for item in raw["references"])
        return cls(**values)


@dataclass(frozen=True)
class FeatureBank:
    metadata: BankMetadata
    features: torch.Tensor


class BankStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for(self, bank_id: str) -> Path:
        _validate_bank_id(bank_id)
        return self.root / bank_id

    def build(
        self,
        bank_id: str,
        reference_paths: Iterable[str | Path],
        *,
        backend: SpeechBackend,
        config: AppConfig,
        overwrite: bool = False,
    ) -> FeatureBank:
        paths = _unique_audio_paths(reference_paths)
        if not paths:
            raise ValueError("At least one target reference recording is required")

        feature_parts: list[torch.Tensor] = []
        records: list[ReferenceRecord] = []
        for path in paths:
            waveform = load_audio(
                path,
                sample_rate=config.audio.sample_rate,
                max_seconds=config.audio.reference_max_seconds,
            )
            features = backend.extract_features(
                waveform,
                vad_trigger_db=config.audio.reference_vad_trigger_db,
            )
            feature_parts.append(features)
            records.append(
                ReferenceRecord(
                    name=path.name,
                    sha256=_sha256_file(path),
                    duration_seconds=round(waveform.shape[1] / config.audio.sample_rate, 6),
                )
            )

        all_features = torch.cat(feature_parts, dim=0).to(torch.float32).contiguous()
        metadata = BankMetadata(
            format_version=BANK_FORMAT_VERSION,
            bank_id=bank_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            model_repository=backend.repository,
            wavlm_sha256=backend.wavlm_sha256,
            vocoder_sha256=backend.vocoder_sha256,
            feature_layer=backend.feature_layer,
            sample_rate=config.audio.sample_rate,
            vad_trigger_db=config.audio.reference_vad_trigger_db,
            reference_count=len(records),
            reference_duration_seconds=round(sum(item.duration_seconds for item in records), 6),
            feature_frames=all_features.shape[0],
            feature_dimension=all_features.shape[1],
            feature_dtype="float32",
            features_sha256="",
            references=tuple(records),
        )
        return self.save(FeatureBank(metadata, all_features), overwrite=overwrite)

    def save(self, bank: FeatureBank, *, overwrite: bool = False) -> FeatureBank:
        destination = self.path_for(bank.metadata.bank_id)
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Feature bank already exists; use overwrite=True to replace it: {destination}"
            )
        self.root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix=f".{bank.metadata.bank_id}-", dir=self.root
        ) as temp:
            temp_path = Path(temp)
            features_path = temp_path / "features.npy"
            np.save(features_path, bank.features.cpu().numpy().astype(np.float32, copy=False))
            checksum = _sha256_file(features_path)
            metadata = BankMetadata(
                **{
                    **asdict(bank.metadata),
                    "features_sha256": checksum,
                    "references": bank.metadata.references,
                }
            )
            with (temp_path / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(asdict(metadata), handle, indent=2)
                handle.write("\n")

            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(temp_path, destination)

        return FeatureBank(metadata=metadata, features=bank.features)

    def load(self, bank_id: str, *, verify_checksum: bool = True) -> FeatureBank:
        bank_path = self.path_for(bank_id)
        metadata_path = bank_path / "metadata.json"
        features_path = bank_path / "features.npy"
        if not metadata_path.is_file() or not features_path.is_file():
            raise FileNotFoundError(
                f"Feature bank '{bank_id}' is incomplete or missing under {self.root}"
            )

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = BankMetadata.from_dict(json.load(handle))
        if metadata.format_version != BANK_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported bank format {metadata.format_version}; expected {BANK_FORMAT_VERSION}"
            )
        if metadata.bank_id != bank_id:
            raise ValueError(f"Bank metadata ID '{metadata.bank_id}' does not match '{bank_id}'")
        if verify_checksum and _sha256_file(features_path) != metadata.features_sha256:
            raise ValueError(f"Feature checksum verification failed for bank '{bank_id}'")

        array = np.load(features_path, allow_pickle=False)
        features = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
        expected_shape = (metadata.feature_frames, metadata.feature_dimension)
        if tuple(features.shape) != expected_shape:
            raise ValueError(
                f"Feature shape {tuple(features.shape)} does not match metadata {expected_shape}"
            )
        return FeatureBank(metadata=metadata, features=features)

    def list(self) -> list[BankMetadata]:
        if not self.root.is_dir():
            return []
        banks: list[BankMetadata] = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or not _BANK_ID.fullmatch(path.name):
                continue
            try:
                banks.append(self.load(path.name, verify_checksum=False).metadata)
            except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return banks


def validate_bank_compatibility(
    bank: FeatureBank, backend: SpeechBackend, config: AppConfig
) -> None:
    errors: list[str] = []
    metadata = bank.metadata
    if metadata.model_repository != backend.repository:
        errors.append(
            f"model repository differs ({metadata.model_repository} != {backend.repository})"
        )
    if metadata.wavlm_sha256 != backend.wavlm_sha256:
        errors.append("WavLM checkpoint fingerprint differs")
    if metadata.vocoder_sha256 != backend.vocoder_sha256:
        errors.append("vocoder checkpoint fingerprint differs")
    if metadata.feature_layer != backend.feature_layer:
        errors.append(
            f"feature layer differs ({metadata.feature_layer} != {backend.feature_layer})"
        )
    if metadata.sample_rate != config.audio.sample_rate:
        errors.append(f"sample rate differs ({metadata.sample_rate} != {config.audio.sample_rate})")
    if errors:
        raise ValueError(f"Feature bank '{metadata.bank_id}' is incompatible: " + "; ".join(errors))


def _validate_bank_id(bank_id: str) -> None:
    if not _BANK_ID.fullmatch(bank_id):
        raise ValueError(
            "Bank ID must start with an alphanumeric character and contain only "
            "letters, numbers, periods, underscores, or hyphens (maximum 64 characters)"
        )


def _unique_audio_paths(paths: Iterable[str | Path]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Reference audio file does not exist: {path}")
        if path not in seen:
            seen.add(path)
            resolved.append(path)
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
