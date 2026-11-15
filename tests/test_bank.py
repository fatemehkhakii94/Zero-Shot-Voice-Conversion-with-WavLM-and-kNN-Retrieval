from pathlib import Path

import pytest
from conftest import FakeBackend, write_test_audio

from knnvc.bank import BankStore, validate_bank_compatibility
from knnvc.config import AppConfig, AudioConfig, ModelConfig, PathConfig


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        model=ModelConfig(repository=FakeBackend.repository, device="cpu"),
        audio=AudioConfig(reference_vad_trigger_db=0.0),
        paths=PathConfig(bank_dir=str(tmp_path / "banks")),
    )


def test_bank_round_trip_with_checksum(tmp_path: Path, fake_backend: FakeBackend) -> None:
    config = _config(tmp_path)
    reference = write_test_audio(tmp_path / "reference.wav")
    store = BankStore(config.paths.bank_dir)

    built = store.build("speaker-1", [reference], backend=fake_backend, config=config)
    loaded = store.load("speaker-1")

    assert loaded.metadata.reference_count == 1
    assert loaded.metadata.features_sha256 == built.metadata.features_sha256
    assert loaded.features.shape[1] == 4
    validate_bank_compatibility(loaded, fake_backend, config)


def test_bank_refuses_implicit_overwrite(tmp_path: Path, fake_backend: FakeBackend) -> None:
    config = _config(tmp_path)
    reference = write_test_audio(tmp_path / "reference.wav")
    store = BankStore(config.paths.bank_dir)
    store.build("speaker-1", [reference], backend=fake_backend, config=config)
    with pytest.raises(FileExistsError):
        store.build("speaker-1", [reference], backend=fake_backend, config=config)


def test_bank_detects_feature_tampering(tmp_path: Path, fake_backend: FakeBackend) -> None:
    config = _config(tmp_path)
    reference = write_test_audio(tmp_path / "reference.wav")
    store = BankStore(config.paths.bank_dir)
    store.build("speaker-1", [reference], backend=fake_backend, config=config)
    features = Path(config.paths.bank_dir) / "speaker-1" / "features.npy"
    with features.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        store.load("speaker-1")
