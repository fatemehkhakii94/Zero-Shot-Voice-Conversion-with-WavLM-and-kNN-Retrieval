from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MANIFEST_FORMAT_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class BankDefinition:
    bank_id: str
    references: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationPair:
    pair_id: str
    source: str
    transcript: str
    target_bank: str
    target_references: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationManifest:
    format_version: int
    dataset: str
    banks: tuple[BankDefinition, ...]
    pairs: tuple[EvaluationPair, ...]
    sha256: str

    def bank(self, bank_id: str) -> BankDefinition:
        for bank in self.banks:
            if bank.bank_id == bank_id:
                return bank
        raise KeyError(f"Manifest does not define target bank '{bank_id}'")


def load_manifest(path: str | Path) -> EvaluationManifest:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest does not exist: {manifest_path}")
    content = manifest_path.read_bytes()
    raw = yaml.safe_load(content) or {}
    if not isinstance(raw, dict):
        raise ValueError("Evaluation manifest root must be a mapping")
    _expect_keys(raw, {"format_version", "dataset", "banks", "pairs"}, "manifest")
    if raw.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported manifest format {raw.get('format_version')}; "
            f"expected {MANIFEST_FORMAT_VERSION}"
        )

    bank_items = raw.get("banks")
    pair_items = raw.get("pairs")
    if not isinstance(bank_items, list) or not bank_items:
        raise ValueError("Evaluation manifest must contain a non-empty banks list")
    if not isinstance(pair_items, list) or not pair_items:
        raise ValueError("Evaluation manifest must contain a non-empty pairs list")

    banks: list[BankDefinition] = []
    for index, item in enumerate(bank_items):
        if not isinstance(item, dict):
            raise ValueError(f"banks[{index}] must be a mapping")
        _expect_keys(item, {"bank_id", "references"}, f"banks[{index}]")
        bank_id = _identifier(item.get("bank_id"), f"banks[{index}].bank_id")
        references = _string_tuple(item.get("references"), f"banks[{index}].references")
        banks.append(BankDefinition(bank_id=bank_id, references=references))

    bank_ids = [bank.bank_id for bank in banks]
    if len(set(bank_ids)) != len(bank_ids):
        raise ValueError("Evaluation manifest contains duplicate bank IDs")

    pairs: list[EvaluationPair] = []
    for index, item in enumerate(pair_items):
        if not isinstance(item, dict):
            raise ValueError(f"pairs[{index}] must be a mapping")
        _expect_keys(
            item,
            {"pair_id", "source", "transcript", "target_bank", "target_references"},
            f"pairs[{index}]",
        )
        pair = EvaluationPair(
            pair_id=_identifier(item.get("pair_id"), f"pairs[{index}].pair_id"),
            source=_string(item.get("source"), f"pairs[{index}].source"),
            transcript=_string(item.get("transcript"), f"pairs[{index}].transcript"),
            target_bank=_identifier(item.get("target_bank"), f"pairs[{index}].target_bank"),
            target_references=_string_tuple(
                item.get("target_references"), f"pairs[{index}].target_references"
            ),
        )
        if pair.target_bank not in bank_ids:
            raise ValueError(f"Pair '{pair.pair_id}' refers to undefined bank '{pair.target_bank}'")
        pairs.append(pair)

    pair_ids = [pair.pair_id for pair in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("Evaluation manifest contains duplicate pair IDs")

    bank_references = {bank.bank_id: set(bank.references) for bank in banks}
    for pair in pairs:
        overlap = bank_references[pair.target_bank].intersection(pair.target_references)
        if overlap:
            raise ValueError(
                f"Pair '{pair.pair_id}' leaks held-out target references into its bank: "
                + ", ".join(sorted(overlap))
            )

    return EvaluationManifest(
        format_version=MANIFEST_FORMAT_VERSION,
        dataset=_string(raw.get("dataset"), "dataset"),
        banks=tuple(banks),
        pairs=tuple(pairs),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _expect_keys(raw: dict[str, Any], expected: set[str], location: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    errors: list[str] = []
    if missing:
        errors.append(f"missing {', '.join(missing)}")
    if unknown:
        errors.append(f"unknown {', '.join(unknown)}")
    if errors:
        raise ValueError(f"Invalid {location}: {'; '.join(errors)}")


def _identifier(value: Any, location: str) -> str:
    text = _string(value, location)
    if not _IDENTIFIER.fullmatch(text):
        raise ValueError(f"{location} is not a valid identifier: {text!r}")
    return text


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value.strip()


def _string_tuple(value: Any, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty list")
    result = tuple(_string(item, f"{location}[]") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{location} contains duplicate paths")
    return result
