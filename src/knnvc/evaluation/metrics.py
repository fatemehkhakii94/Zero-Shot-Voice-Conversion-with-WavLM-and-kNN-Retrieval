from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class WordErrorScore:
    errors: int
    reference_words: int
    wer: float
    reference: str
    hypothesis: str


def normalize_transcript(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    return " ".join(text.split())


def word_error_score(reference: str, hypothesis: str) -> WordErrorScore:
    normalized_reference = normalize_transcript(reference)
    normalized_hypothesis = normalize_transcript(hypothesis)
    reference_words = normalized_reference.split()
    hypothesis_words = normalized_hypothesis.split()
    errors = _edit_distance(reference_words, hypothesis_words)
    denominator = len(reference_words)
    wer = errors / denominator if denominator else float(bool(hypothesis_words))
    return WordErrorScore(
        errors=errors,
        reference_words=denominator,
        wer=wer,
        reference=normalized_reference,
        hypothesis=normalized_hypothesis,
    )


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_word in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_word in enumerate(hypothesis, start=1):
            substitution = previous[column - 1] + (reference_word != hypothesis_word)
            current.append(min(previous[column] + 1, current[column - 1] + 1, substitution))
        previous = current
    return previous[-1]


class WhisperTranscriber:
    def __init__(self, model_name: str, device: torch.device, language: str = "en"):
        try:
            import whisper
        except ImportError as error:
            raise RuntimeError(
                "Whisper evaluation is not installed; install knnvc-toolkit[evaluation]"
            ) from error
        self._model: Any = whisper.load_model(model_name, device=str(device))
        self.device = device
        self.language = language

    def transcribe(self, path: str | Path) -> str:
        result = self._model.transcribe(
            str(path),
            language=self.language,
            fp16=self.device.type == "cuda",
        )
        return normalize_transcript(str(result["text"]))


class SpeakerSimilarityScorer:
    def __init__(self, device: torch.device):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as error:
            raise RuntimeError(
                "SpeechBrain evaluation is not installed; install knnvc-toolkit[evaluation]"
            ) from error
        self.device = device
        run_device = f"cuda:{device.index or 0}" if device.type == "cuda" else str(device)
        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": run_device},
        )

    @torch.inference_mode()
    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        embedding = self._model.encode_batch(waveform.to(self.device))
        return functional.normalize(embedding.squeeze(1), p=2, dim=-1).cpu()

    @staticmethod
    def similarity(first: torch.Tensor, second: torch.Tensor) -> float:
        return float(functional.cosine_similarity(first, second, dim=-1).mean().item())


class DNSMOSScorer:
    def __init__(self) -> None:
        try:
            from torchmetrics.functional.audio.dnsmos import (
                deep_noise_suppression_mean_opinion_score,
            )
        except ImportError as error:
            raise RuntimeError(
                "DNSMOS evaluation is not installed; install knnvc-toolkit[evaluation]"
            ) from error
        self._metric = deep_noise_suppression_mean_opinion_score

    def score(self, path: str | Path) -> dict[str, float]:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(path))
        scores = (
            self._metric(
                waveform.squeeze(0),
                fs=sample_rate,
                personalized=False,
                device="cpu",
            )
            .detach()
            .cpu()
        )
        return {
            "p808": round(float(scores[0]), 6),
            "signal": round(float(scores[1]), 6),
            "background": round(float(scores[2]), 6),
            "overall": round(float(scores[3]), 6),
        }
