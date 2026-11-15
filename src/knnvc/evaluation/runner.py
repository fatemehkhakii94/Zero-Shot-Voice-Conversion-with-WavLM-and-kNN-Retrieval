from __future__ import annotations

import json
import platform
import re
import statistics
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from knnvc import __version__
from knnvc.audio import load_audio
from knnvc.config import AppConfig, resolve_device
from knnvc.converter import VoiceConverter
from knnvc.evaluation.manifest import EvaluationManifest
from knnvc.evaluation.metrics import (
    DNSMOSScorer,
    SpeakerSimilarityScorer,
    WhisperTranscriber,
    word_error_score,
)


class EvaluationRunner:
    def __init__(
        self,
        config: AppConfig,
        manifest: EvaluationManifest,
        dataset_root: str | Path,
        *,
        converter: VoiceConverter | None = None,
        transcriber: WhisperTranscriber | None = None,
        speaker_scorer: SpeakerSimilarityScorer | None = None,
        quality_scorer: DNSMOSScorer | None = None,
    ):
        self.config = config
        self.manifest = manifest
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.converter = converter or VoiceConverter(config)
        evaluation_device = resolve_device(config.model.device)
        self.transcriber = transcriber or WhisperTranscriber(
            config.evaluation.whisper_model,
            evaluation_device,
            config.evaluation.language,
        )
        self.speaker_scorer = speaker_scorer or SpeakerSimilarityScorer(evaluation_device)
        self.quality_scorer = quality_scorer or DNSMOSScorer()
        self._embedding_cache: dict[Path, torch.Tensor] = {}

    def run(self, output_path: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
        report_path = Path(output_path)
        if report_path.exists() and not overwrite:
            raise FileExistsError(f"Evaluation report already exists: {report_path}")
        dataset_slug = _slug(self.manifest.dataset, fallback=self.manifest.sha256[:12])
        audio_dir = Path(self.config.paths.audio_dir) / "evaluation" / dataset_slug
        report_path.parent.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        for pair in tqdm(self.manifest.pairs, desc="Evaluation pairs"):
            source_path = self._dataset_path(pair.source)
            target_paths = [self._dataset_path(value) for value in pair.target_references]
            output_audio = audio_dir / f"{pair.pair_id}.wav"

            for _ in range(self.config.evaluation.timing_warmup_runs):
                self.converter.convert(
                    source_path,
                    pair.target_bank,
                    output_audio,
                    overwrite=True,
                    write_metadata=False,
                )

            timing_results = [
                self.converter.convert(
                    source_path,
                    pair.target_bank,
                    output_audio,
                    overwrite=True,
                    write_metadata=False,
                )
                for _ in range(self.config.evaluation.timing_runs)
            ]
            representative = timing_results[-1]
            inference_times = [item.inference_seconds for item in timing_results]
            median_inference = statistics.median(inference_times)
            rtf = median_inference / representative.source_duration_seconds

            source_text = self.transcriber.transcribe(source_path)
            converted_text = self.transcriber.transcribe(output_audio)
            source_wer = word_error_score(pair.transcript, source_text)
            converted_wer = word_error_score(pair.transcript, converted_text)

            source_waveform = load_audio(source_path, sample_rate=self.config.audio.sample_rate)
            converted_waveform = load_audio(output_audio, sample_rate=self.config.audio.sample_rate)
            source_embedding = self.speaker_scorer.embed(source_waveform)
            converted_embedding = self.speaker_scorer.embed(converted_waveform)
            target_embeddings = [self._target_embedding(path) for path in target_paths]
            source_target = statistics.mean(
                self.speaker_scorer.similarity(source_embedding, target)
                for target in target_embeddings
            )
            converted_target = statistics.mean(
                self.speaker_scorer.similarity(converted_embedding, target)
                for target in target_embeddings
            )
            converted_source = self.speaker_scorer.similarity(converted_embedding, source_embedding)
            conversion = representative.to_dict()
            conversion["source"] = pair.source
            conversion["output"] = output_audio.as_posix()

            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "source": pair.source,
                    "target_bank": pair.target_bank,
                    "target_references": list(pair.target_references),
                    "output": output_audio.as_posix(),
                    "speaker_similarity": {
                        "source_to_target": round(source_target, 6),
                        "converted_to_target": round(converted_target, 6),
                        "converted_to_source": round(converted_source, 6),
                        "target_gain": round(converted_target - source_target, 6),
                    },
                    "transcription": {
                        "ground_truth": converted_wer.reference,
                        "source": source_wer.hypothesis,
                        "converted": converted_wer.hypothesis,
                        "source_errors": source_wer.errors,
                        "converted_errors": converted_wer.errors,
                        "reference_words": converted_wer.reference_words,
                        "source_wer": round(source_wer.wer, 6),
                        "converted_wer": round(converted_wer.wer, 6),
                    },
                    "dnsmos": {
                        "source": self.quality_scorer.score(source_path),
                        "converted": self.quality_scorer.score(output_audio),
                    },
                    "timing": {
                        "runs": len(inference_times),
                        "inference_seconds": inference_times,
                        "median_inference_seconds": round(median_inference, 6),
                        "real_time_factor": round(rtf, 6),
                    },
                    "conversion": conversion,
                }
            )

        report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": self.manifest.dataset,
            "manifest_sha256": self.manifest.sha256,
            "environment": _environment(),
            "config": asdict(self.config),
            "summary": _summarize(rows),
            "pairs": rows,
        }
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        return report

    def _dataset_path(self, relative_path: str) -> Path:
        candidate = (self.dataset_root / relative_path).resolve()
        try:
            candidate.relative_to(self.dataset_root)
        except ValueError as error:
            raise ValueError(f"Manifest path escapes the dataset root: {relative_path}") from error
        if not candidate.is_file():
            raise FileNotFoundError(f"Manifest audio file does not exist: {candidate}")
        return candidate

    def _target_embedding(self, path: Path) -> torch.Tensor:
        if path not in self._embedding_cache:
            waveform = load_audio(path, sample_rate=self.config.audio.sample_rate)
            self._embedding_cache[path] = self.speaker_scorer.embed(waveform)
        return self._embedding_cache[path]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    converted_target = [row["speaker_similarity"]["converted_to_target"] for row in rows]
    source_target = [row["speaker_similarity"]["source_to_target"] for row in rows]
    target_gain = [row["speaker_similarity"]["target_gain"] for row in rows]
    converted_dns = [row["dnsmos"]["converted"]["overall"] for row in rows]
    source_dns = [row["dnsmos"]["source"]["overall"] for row in rows]
    rtf = [row["timing"]["real_time_factor"] for row in rows]
    converted_errors = sum(row["transcription"]["converted_errors"] for row in rows)
    source_errors = sum(row["transcription"]["source_errors"] for row in rows)
    reference_words = sum(row["transcription"]["reference_words"] for row in rows)
    macro_wer = [row["transcription"]["converted_wer"] for row in rows]

    return {
        "pair_count": len(rows),
        "speaker_similarity": {
            "converted_to_target": _mean_with_ci(converted_target),
            "source_to_target": _mean_with_ci(source_target),
            "target_gain": _mean_with_ci(target_gain),
        },
        "wer": {
            "converted_corpus": round(converted_errors / reference_words, 6),
            "source_corpus": round(source_errors / reference_words, 6),
            "converted_macro": round(statistics.mean(macro_wer), 6),
            "converted_errors": converted_errors,
            "source_errors": source_errors,
            "reference_words": reference_words,
        },
        "dnsmos_overall": {
            "converted": _mean_with_ci(converted_dns),
            "source": _mean_with_ci(source_dns),
        },
        "real_time_factor": {
            "median": round(statistics.median(rtf), 6),
            "mean": round(statistics.mean(rtf), 6),
            "p95": round(float(np.percentile(rtf, 95)), 6),
        },
    }


def _mean_with_ci(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) == 1:
        return {"mean": round(mean, 6), "ci95_low": round(mean, 6), "ci95_high": round(mean, 6)}
    generator = np.random.default_rng(0)
    samples = generator.choice(array, size=(2_000, len(array)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "mean": round(mean, 6),
        "ci95_low": round(float(low), 6),
        "ci95_high": round(float(high), 6),
    }


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in (
        "numpy",
        "openai-whisper",
        "speechbrain",
        "torch",
        "torchaudio",
        "torchmetrics",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "knnvc_toolkit": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": cuda_name,
        "packages": packages,
    }


def _slug(value: str, *, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.casefold()).strip("-._")
    return slug or fallback
