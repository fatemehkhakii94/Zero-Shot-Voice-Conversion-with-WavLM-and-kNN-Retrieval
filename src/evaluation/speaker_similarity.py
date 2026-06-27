"""
Speaker similarity using ECAPA-TDNN embeddings (SpeechBrain).
Cosine similarity between converted audio and target speaker reference.
"""
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier

from src.audio.preprocessing import load_audio


class SpeakerSimilarity:
    def __init__(self, device: str = "cuda"):
        self.device = device
        print("Loading ECAPA-TDNN ...")
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        print("ECAPA-TDNN ready.")

    @torch.inference_mode()
    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (1, T) at 16kHz → (1, 192) embedding"""
        wav = wav.to(self.device)
        emb = self.model.encode_batch(wav)  # (1, 1, 192)
        return emb.squeeze(1)               # (1, 192)

    def score(self, converted_wav: torch.Tensor, target_wav: torch.Tensor) -> float:
        """Returns cosine similarity in [-1, 1]. Higher = more similar."""
        emb_conv   = self.embed(converted_wav)
        emb_target = self.embed(target_wav)
        sim = F.cosine_similarity(emb_conv, emb_target, dim=-1)
        return sim.item()
