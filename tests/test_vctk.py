from pathlib import Path

from conftest import write_test_audio

from knnvc.evaluation.manifest import load_manifest
from knnvc.vctk import prepare_vctk_manifest


def test_vctk_manifest_has_disjoint_target_references(tmp_path: Path) -> None:
    for speaker in ("p001", "p002"):
        for index in range(1, 7):
            stem = f"{speaker}_{index:03d}"
            write_test_audio(tmp_path / "wav48_silence_trimmed" / speaker / f"{stem}_mic1.flac")
            text = tmp_path / "txt" / speaker / f"{stem}.txt"
            text.parent.mkdir(parents=True, exist_ok=True)
            text.write_text(f"sentence {index}", encoding="utf-8")

    output = tmp_path / "manifest.yaml"
    prepare_vctk_manifest(
        tmp_path,
        output,
        max_speakers=2,
        pairs_per_speaker=2,
        heldout_references=1,
        bank_references=3,
    )
    manifest = load_manifest(output)

    assert len(manifest.banks) == 2
    assert len(manifest.pairs) == 4
    for pair in manifest.pairs:
        assert set(pair.target_references).isdisjoint(manifest.bank(pair.target_bank).references)
