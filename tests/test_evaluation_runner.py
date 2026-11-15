from __future__ import annotations

from pathlib import Path

import torch
from conftest import FakeBackend, write_test_audio

from knnvc.bank import BankStore
from knnvc.config import (
    AppConfig,
    AudioConfig,
    EvaluationConfig,
    MatchingConfig,
    ModelConfig,
    PathConfig,
)
from knnvc.converter import VoiceConverter
from knnvc.evaluation.manifest import load_manifest
from knnvc.evaluation.runner import EvaluationRunner


class FakeTranscriber:
    def transcribe(self, path: str | Path) -> str:
        del path
        return "hello world"


class FakeSpeakerScorer:
    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        mean = waveform.mean()
        return torch.tensor([[float(mean), 1.0]])

    @staticmethod
    def similarity(first: torch.Tensor, second: torch.Tensor) -> float:
        return float(torch.nn.functional.cosine_similarity(first, second).item())


class FakeQualityScorer:
    def score(self, path: str | Path) -> dict[str, float]:
        del path
        return {"overall": 3.5, "signal": 3.6, "background": 4.0}


def test_evaluation_report_contains_corpus_metrics(tmp_path: Path) -> None:
    config = AppConfig(
        model=ModelConfig(repository=FakeBackend.repository, device="cpu"),
        audio=AudioConfig(reference_vad_trigger_db=0.0),
        matching=MatchingConfig(k=1, device="cpu"),
        paths=PathConfig(
            bank_dir=str(tmp_path / "banks"),
            audio_dir=str(tmp_path / "audio"),
            report_dir=str(tmp_path / "reports"),
        ),
        evaluation=EvaluationConfig(timing_warmup_runs=0, timing_runs=1),
    )
    source = write_test_audio(tmp_path / "dataset" / "source.wav")
    bank_reference = write_test_audio(tmp_path / "dataset" / "bank.wav")
    target_reference = write_test_audio(tmp_path / "dataset" / "target.wav")
    store = BankStore(config.paths.bank_dir)
    backend = FakeBackend()
    store.build("target", [bank_reference], backend=backend, config=config)
    converter = VoiceConverter(config, backend=backend, bank_store=store)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        """\
format_version: 1
dataset: fixture
banks:
  - bank_id: target
    references: [bank.wav]
pairs:
  - pair_id: source-to-target
    source: source.wav
    transcript: hello world
    target_bank: target
    target_references: [target.wav]
""",
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    runner = EvaluationRunner(
        config,
        manifest,
        tmp_path / "dataset",
        converter=converter,
        transcriber=FakeTranscriber(),  # type: ignore[arg-type]
        speaker_scorer=FakeSpeakerScorer(),  # type: ignore[arg-type]
        quality_scorer=FakeQualityScorer(),  # type: ignore[arg-type]
    )

    report = runner.run(tmp_path / "report.json")

    assert report["summary"]["pair_count"] == 1
    assert report["summary"]["wer"]["converted_corpus"] == 0.0
    assert Path(report["pairs"][0]["output"]).is_file()
    assert report["pairs"][0]["conversion"]["source"] == "source.wav"
    assert report["pairs"][0]["conversion"]["output"] == report["pairs"][0]["output"]
    assert "evaluation/fixture" in report["pairs"][0]["output"].replace("\\", "/")
    assert source.is_file() and target_reference.is_file()
