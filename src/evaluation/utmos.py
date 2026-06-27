"""
UTMOS: automatic MOS prediction (speech quality).
Uses speechmos library.
"""
import torch
import torchaudio
from pathlib import Path
from speechmos import dnsmos


class UTMOSScorer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        print("UTMOS (DNSMOS) ready — no download required.")

    def score(self, audio_path: str | Path) -> dict:
        """
        Returns DNSMOS scores: ovrl, sig, bak.
        ovrl = overall quality (main metric, ~1-5 scale).
        """
        result = dnsmos.run(str(audio_path), sr=16000)
        return {
            "ovrl": round(result["ovrl_mos"], 4),
            "sig":  round(result["sig_mos"],  4),
            "bak":  round(result["bak_mos"],  4),
        }
