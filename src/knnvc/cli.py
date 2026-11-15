from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from knnvc.backend import KNNVCBackend
from knnvc.bank import BankStore
from knnvc.config import AppConfig, load_config
from knnvc.converter import VoiceConverter
from knnvc.evaluation.manifest import load_manifest
from knnvc.evaluation.runner import EvaluationRunner
from knnvc.vctk import prepare_vctk_manifest

LOGGER = logging.getLogger("knnvc")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knnvc",
        description="Training-free voice conversion with WavLM and kNN retrieval.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="YAML configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    commands = parser.add_subparsers(dest="command", required=True)

    bank = commands.add_parser("bank", help="Create and inspect target-speaker feature banks")
    bank_commands = bank.add_subparsers(dest="bank_command", required=True)

    bank_build = bank_commands.add_parser("build", help="Build one bank from reference audio")
    bank_build.add_argument("--id", required=True, dest="bank_id", help="Stable bank identifier")
    bank_build.add_argument(
        "--reference", nargs="+", required=True, help="One or more target reference recordings"
    )
    bank_build.add_argument("--overwrite", action="store_true")

    bank_manifest = bank_commands.add_parser(
        "build-manifest", help="Build every bank defined by an evaluation manifest"
    )
    bank_manifest.add_argument("--manifest", required=True)
    bank_manifest.add_argument("--dataset-root", required=True)
    bank_manifest.add_argument("--overwrite", action="store_true")

    bank_commands.add_parser("list", help="List available feature banks")

    convert = commands.add_parser("convert", help="Convert one source recording")
    convert.add_argument("--source", required=True)
    convert.add_argument("--target-bank", required=True)
    convert.add_argument("--output")
    convert.add_argument("--overwrite", action="store_true")

    data = commands.add_parser("data", help="Prepare reproducible dataset manifests")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    vctk = data_commands.add_parser("prepare-vctk", help="Create a leakage-safe VCTK manifest")
    vctk.add_argument("--dataset-root", required=True)
    vctk.add_argument("--output", default="data/manifests/vctk-evaluation.yaml")
    vctk.add_argument("--speakers", nargs="+")
    vctk.add_argument("--max-speakers", type=int, default=8)
    vctk.add_argument("--pairs-per-speaker", type=int, default=5)
    vctk.add_argument("--heldout-references", type=int, default=3)
    vctk.add_argument("--bank-references", type=int, default=20)

    evaluate = commands.add_parser("evaluate", help="Run the reproducible evaluation protocol")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--dataset-root", required=True)
    evaluate.add_argument(
        "--output",
        help="Report path (default: <configured report_dir>/evaluation.json)",
    )
    evaluate.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        config = load_config(args.config)
        _dispatch(args, config)
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        LOGGER.error("%s", error)
        if args.verbose:
            LOGGER.exception("Command failed")
        raise SystemExit(2) from error


def _dispatch(args: argparse.Namespace, config: AppConfig) -> None:
    if args.command == "bank":
        _bank_command(args, config)
        return
    if args.command == "convert":
        result = VoiceConverter(config).convert(
            args.source,
            args.target_bank,
            args.output,
            overwrite=args.overwrite,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return
    if args.command == "data" and args.data_command == "prepare-vctk":
        output = prepare_vctk_manifest(
            args.dataset_root,
            args.output,
            speakers=args.speakers,
            max_speakers=args.max_speakers,
            pairs_per_speaker=args.pairs_per_speaker,
            heldout_references=args.heldout_references,
            bank_references=args.bank_references,
        )
        LOGGER.info("Wrote VCTK evaluation manifest to %s", output)
        return
    if args.command == "evaluate":
        manifest = load_manifest(args.manifest)
        report_output = args.output or str(Path(config.paths.report_dir) / "evaluation.json")
        report = EvaluationRunner(config, manifest, args.dataset_root).run(
            report_output,
            overwrite=args.overwrite,
        )
        print(json.dumps(report["summary"], indent=2))
        return
    raise RuntimeError("Unhandled command")


def _bank_command(args: argparse.Namespace, config: AppConfig) -> None:
    store = BankStore(config.paths.bank_dir)
    if args.bank_command == "list":
        print(json.dumps([asdict(item) for item in store.list()], indent=2))
        return

    backend = KNNVCBackend(config)
    if args.bank_command == "build":
        bank = store.build(
            args.bank_id,
            args.reference,
            backend=backend,
            config=config,
            overwrite=args.overwrite,
        )
        print(json.dumps(asdict(bank.metadata), indent=2))
        return

    if args.bank_command == "build-manifest":
        manifest = load_manifest(args.manifest)
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        for definition in tqdm(manifest.banks, desc="Feature banks"):
            references = [_safe_dataset_path(dataset_root, item) for item in definition.references]
            store.build(
                definition.bank_id,
                references,
                backend=backend,
                config=config,
                overwrite=args.overwrite,
            )
        return
    raise RuntimeError("Unhandled bank command")


def _safe_dataset_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Manifest path escapes the dataset root: {value}") from error
    return path


if __name__ == "__main__":
    main(sys.argv[1:])
