from pathlib import Path

import pytest

from knnvc.evaluation.manifest import load_manifest


def _manifest(target_reference: str = "audio/p2/eval.flac") -> str:
    return f"""\
format_version: 1
dataset: test
banks:
  - bank_id: p2
    references:
      - audio/p2/bank.flac
pairs:
  - pair_id: p1-to-p2
    source: audio/p1/source.flac
    transcript: hello world
    target_bank: p2
    target_references:
      - {target_reference}
"""


def test_manifest_loads_and_hashes_content(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(_manifest(), encoding="utf-8")
    manifest = load_manifest(path)
    assert manifest.dataset == "test"
    assert manifest.bank("p2").references == ("audio/p2/bank.flac",)
    assert len(manifest.sha256) == 64


def test_manifest_rejects_reference_leakage(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(_manifest("audio/p2/bank.flac"), encoding="utf-8")
    with pytest.raises(ValueError, match="leaks held-out"):
        load_manifest(path)
