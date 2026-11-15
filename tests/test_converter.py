import json
from pathlib import Path

from conftest import FakeBackend, write_test_audio

from knnvc.bank import BankStore
from knnvc.config import AppConfig, AudioConfig, MatchingConfig, ModelConfig, PathConfig
from knnvc.converter import VoiceConverter


def test_converter_uses_bank_and_writes_metadata(
    tmp_path: Path,
    fake_backend: FakeBackend,
) -> None:
    config = AppConfig(
        model=ModelConfig(repository=FakeBackend.repository, device="cpu"),
        audio=AudioConfig(reference_vad_trigger_db=0.0),
        matching=MatchingConfig(k=1, device="cpu", reference_chunk_size=10),
        paths=PathConfig(
            bank_dir=str(tmp_path / "banks"),
            audio_dir=str(tmp_path / "audio"),
            report_dir=str(tmp_path / "reports"),
        ),
    )
    source = write_test_audio(tmp_path / "source.wav")
    reference = write_test_audio(tmp_path / "reference.wav")
    store = BankStore(config.paths.bank_dir)
    store.build("target", [reference], backend=fake_backend, config=config)
    converter = VoiceConverter(config, backend=fake_backend, bank_store=store)

    result = converter.convert(source, "target")

    output = Path(result.output)
    assert output.is_file()
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["target_bank"] == "target"
    assert metadata["metric"] == "cosine"
    assert metadata["source_frames"] > 0
