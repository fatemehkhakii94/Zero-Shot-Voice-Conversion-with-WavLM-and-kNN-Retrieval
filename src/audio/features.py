import torch
from src.models.model_loader import get_knn_vc


class WavLMFeatureExtractor:
    def __init__(self, cfg: dict):
        self.device = torch.device(cfg["wavlm"]["device"])
        self.layer  = cfg["wavlm"]["content_layer"]
        model       = get_knn_vc(cfg["wavlm"]["device"])
        self.wavlm  = model.wavlm.to(self.device)
        self.wavlm.eval()

    @torch.inference_mode()
    def extract(self, wav: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        wav     : (1, T) float32
        Returns : (frames, 1024) float32 on CPU
        """
        wav = wav.to(self.device)
        features, _ = self.wavlm.extract_features(
            wav,
            output_layer=self.layer,
            ret_layer_results=False,
        )  # (1, frames, 1024)
        return features.squeeze(0).cpu()
