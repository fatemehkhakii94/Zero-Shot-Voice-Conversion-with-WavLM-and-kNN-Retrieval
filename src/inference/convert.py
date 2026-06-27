import torch
import torchaudio
import time
from pathlib import Path

from src.config import load_config
from src.audio.preprocessing import load_audio
from src.audio.features import WavLMFeatureExtractor
from src.models.knn_matcher import KNNMatcher
from src.models.vocoder import HiFiGANVocoder


class VoiceConverter:
    def __init__(self, cfg: dict):
        self.cfg     = cfg
        self.sr      = cfg["audio"]["sample_rate"]
        self.max_dur = cfg["audio"]["max_duration_sec"]
        self.extractor = WavLMFeatureExtractor(cfg)
        self.matcher   = KNNMatcher(
            k=cfg["knn"]["k"],
            use_gpu=cfg["knn"]["use_gpu"],
        )
        self.vocoder = HiFiGANVocoder(cfg)

    def convert(
        self,
        source_path: str | Path,
        target_paths: list[str | Path],
        output_path: str | Path,
    ) -> dict:
        """
        Convert source speaker to target speaker.

        source_path  : single source utterance
        target_paths : one or more reference utterances from target speaker
        output_path  : where to save converted .wav
        Returns      : dict with rtf and shape info
        """
        t_start = time.perf_counter()

        # 1. Load + extract source features
        src_wav  = load_audio(source_path,  self.sr, self.max_dur)
        src_feats = self.extractor.extract(src_wav, self.sr)  # (S, 1024)

        # 2. Load + extract target speaker feature pool
        tgt_feats_list = []
        for p in target_paths:
            w = load_audio(p, self.sr, self.max_dur)
            f = self.extractor.extract(w, self.sr)
            tgt_feats_list.append(f)
        tgt_feats = torch.cat(tgt_feats_list, dim=0)          # (T, 1024)

        # 3. kNN match
        matched = self.matcher.match(src_feats, tgt_feats)    # (S, 1024)

        # 4. Vocode
        out_wav = self.vocoder.vocode(matched)                 # (1, samples)

        t_end = time.perf_counter()

        # 5. Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), out_wav, self.sr)

        audio_dur = src_wav.shape[-1] / self.sr
        rtf       = (t_end - t_start) / audio_dur

        return {
            "source_frames":  src_feats.shape[0],
            "target_frames":  tgt_feats.shape[0],
            "output_samples": out_wav.shape[-1],
            "audio_duration": round(audio_dur, 3),
            "inference_time": round(t_end - t_start, 3),
            "rtf":            round(rtf, 4),
            "output_path":    str(output_path),
        }
