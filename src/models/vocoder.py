import torch
from src.models.model_loader import get_knn_vc


class HiFiGANVocoder:
    def __init__(self, cfg: dict):
        self.device = torch.device(cfg["vocoder"]["device"])
        model       = get_knn_vc(cfg["vocoder"]["device"])
        self.model  = model.hifigan.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def vocode(self, features: torch.Tensor) -> torch.Tensor:
        """
        features : (frames, 1024) float32 on CPU
        Returns  : (1, T) float32 audio at 16kHz on CPU
        """
        x   = features.unsqueeze(0).to(self.device)  # (1, frames, 1024)
        wav = self.model(x)                           # (1, 1, T) or (1, T)
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        return wav.cpu()
