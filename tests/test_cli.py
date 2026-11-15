from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeBackend, write_test_audio

from knnvc import cli
from knnvc.cli import build_parser
from knnvc.config import AppConfig, AudioConfig, ModelConfig, PathConfig


def test_convert_command_parses() -> None:
    args = build_parser().parse_args(
        ["convert", "--source", "source.wav", "--target-bank", "target"]
    )
    assert args.command == "convert"
    assert args.target_bank == "target"


def test_nested_bank_command_parses() -> None:
    args = build_parser().parse_args(
        ["bank", "build", "--id", "target", "--reference", "a.wav", "b.wav"]
    )
    assert args.bank_command == "build"
    assert args.reference == ["a.wav", "b.wav"]


def test_bank_commands_use_one_store(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    config = AppConfig(
        model=ModelConfig(repository=FakeBackend.repository, device="cpu"),
        audio=AudioConfig(reference_vad_trigger_db=0.0),
        paths=PathConfig(bank_dir=str(tmp_path / "banks")),
    )
    monkeypatch.setattr(cli, "KNNVCBackend", lambda config: FakeBackend())
    reference = write_test_audio(tmp_path / "reference.wav")
    cli._bank_command(
        Namespace(bank_command="build", bank_id="target", reference=[reference], overwrite=False),
        config,
    )
    assert '"bank_id": "target"' in capsys.readouterr().out

    cli._bank_command(Namespace(bank_command="list"), config)
    assert '"bank_id": "target"' in capsys.readouterr().out


def test_dispatch_convert_and_data_commands(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    class Result:
        def to_dict(self) -> dict[str, str]:
            return {"status": "ok"}

    class Converter:
        def __init__(self, config: AppConfig):
            del config

        def convert(self, *args: Any, **kwargs: Any) -> Result:
            return Result()

    monkeypatch.setattr(cli, "VoiceConverter", Converter)
    config = AppConfig(model=ModelConfig(device="cpu"))
    cli._dispatch(
        Namespace(
            command="convert",
            source="source.wav",
            target_bank="target",
            output=None,
            overwrite=False,
        ),
        config,
    )
    assert '"status": "ok"' in capsys.readouterr().out

    output = tmp_path / "manifest.yaml"
    monkeypatch.setattr(cli, "prepare_vctk_manifest", lambda *args, **kwargs: output)
    cli._dispatch(
        Namespace(
            command="data",
            data_command="prepare-vctk",
            dataset_root="dataset",
            output=str(output),
            speakers=None,
            max_speakers=2,
            pairs_per_speaker=1,
            heldout_references=1,
            bank_references=1,
        ),
        config,
    )


def test_evaluate_uses_configured_report_directory(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    expected_output = tmp_path / "reports" / "evaluation.json"
    captured: dict[str, Any] = {}

    class Runner:
        def __init__(self, config: AppConfig, manifest: object, dataset_root: str):
            del config, manifest, dataset_root

        def run(self, output: str, *, overwrite: bool) -> dict[str, dict[str, str]]:
            captured.update(output=output, overwrite=overwrite)
            return {"summary": {"status": "ok"}}

    monkeypatch.setattr(cli, "load_manifest", lambda path: object())
    monkeypatch.setattr(cli, "EvaluationRunner", Runner)
    config = AppConfig(paths=PathConfig(report_dir=str(expected_output.parent)))
    cli._dispatch(
        Namespace(
            command="evaluate",
            manifest="manifest.yaml",
            dataset_root="dataset",
            output=None,
            overwrite=False,
        ),
        config,
    )

    assert captured == {"output": str(expected_output), "overwrite": False}
    assert '"status": "ok"' in capsys.readouterr().out


def test_main_reports_configuration_errors(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("model:\n  feature_layer: 5\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--config", str(config), "bank", "list"])
    assert exit_info.value.code == 2
