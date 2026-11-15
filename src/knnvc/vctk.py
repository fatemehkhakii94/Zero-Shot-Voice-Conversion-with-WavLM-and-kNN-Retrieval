from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from knnvc.evaluation.manifest import (
    MANIFEST_FORMAT_VERSION,
    BankDefinition,
    EvaluationPair,
)


def prepare_vctk_manifest(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    speakers: list[str] | None = None,
    max_speakers: int = 8,
    pairs_per_speaker: int = 5,
    heldout_references: int = 3,
    bank_references: int = 20,
) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    audio_root = root / "wav48_silence_trimmed"
    text_root = root / "txt"
    if not audio_root.is_dir() or not text_root.is_dir():
        raise FileNotFoundError(
            f"Expected VCTK directories 'wav48_silence_trimmed' and 'txt' under {root}"
        )
    if pairs_per_speaker < 1 or heldout_references < 1 or bank_references < 1:
        raise ValueError("VCTK split counts must all be positive")

    available = sorted(path.name for path in audio_root.iterdir() if path.is_dir())
    selected = speakers if speakers else available[:max_speakers]
    if len(selected) < 2:
        raise ValueError("At least two VCTK speakers are required for any-to-any evaluation")
    missing = sorted(set(selected) - set(available))
    if missing:
        raise FileNotFoundError(f"VCTK speakers not found: {', '.join(missing)}")

    required = pairs_per_speaker + heldout_references + bank_references
    partitions: dict[str, tuple[list[Path], list[Path], list[Path]]] = {}
    for speaker in selected:
        utterances = sorted((audio_root / speaker).glob(f"{speaker}_*_mic1.flac"))
        if len(utterances) < required:
            raise ValueError(
                f"Speaker {speaker} has {len(utterances)} mic1 recordings; {required} are required"
            )
        sources = utterances[:pairs_per_speaker]
        target_refs = utterances[pairs_per_speaker : pairs_per_speaker + heldout_references]
        bank_refs = utterances[pairs_per_speaker + heldout_references : required]
        partitions[speaker] = (sources, target_refs, bank_refs)

    banks = [
        BankDefinition(
            bank_id=speaker,
            references=tuple(_relative(path, root) for path in partitions[speaker][2]),
        )
        for speaker in selected
    ]

    pairs: list[EvaluationPair] = []
    for source_index, source_speaker in enumerate(selected):
        target_speaker = selected[(source_index + 1) % len(selected)]
        target_references = tuple(_relative(path, root) for path in partitions[target_speaker][1])
        for source_path in partitions[source_speaker][0]:
            transcript_path = (
                text_root / source_speaker / (source_path.stem.removesuffix("_mic1") + ".txt")
            )
            if not transcript_path.is_file():
                raise FileNotFoundError(f"VCTK transcript does not exist: {transcript_path}")
            transcript = transcript_path.read_text(encoding="utf-8").strip()
            pairs.append(
                EvaluationPair(
                    pair_id=f"{source_path.stem}--to--{target_speaker}",
                    source=_relative(source_path, root),
                    transcript=transcript,
                    target_bank=target_speaker,
                    target_references=target_references,
                )
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "dataset": "VCTK 0.92",
        "banks": [asdict(bank) for bank in banks],
        "pairs": [asdict(pair) for pair in pairs],
    }
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return output


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
