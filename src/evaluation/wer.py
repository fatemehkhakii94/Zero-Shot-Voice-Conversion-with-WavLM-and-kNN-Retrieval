"""
Word Error Rate using Whisper.
Measures content preservation: source transcript vs converted transcript.
"""
import whisper
import torchaudio
import torch
from pathlib import Path


def _normalise(text: str) -> str:
    return text.lower().strip()


def _edit_distance(r: list, h: list) -> int:
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(r)][len(h)]


class WERScorer:
    def __init__(self, model_size: str = "base.en", device: str = "cuda"):
        print(f"Loading Whisper {model_size} ...")
        self.model  = whisper.load_model(model_size, device=device)
        self.device = device
        print("Whisper ready.")

    def transcribe(self, audio_path: str | Path) -> str:
        result = self.model.transcribe(str(audio_path), language="en", fp16=True)
        return _normalise(result["text"])

    def score(self, source_path: str | Path, converted_path: str | Path) -> dict:
        """Returns WER and both transcripts."""
        ref = self.transcribe(source_path).split()
        hyp = self.transcribe(converted_path).split()
        if len(ref) == 0:
            return {"wer": 0.0, "reference": "", "hypothesis": ""}
        wer = _edit_distance(ref, hyp) / len(ref)
        return {
            "wer":        round(wer, 4),
            "reference":  " ".join(ref),
            "hypothesis": " ".join(hyp),
        }
