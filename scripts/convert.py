"""
Run voice conversion using precomputed features.
Usage:
    PYTHONPATH=. python scripts/convert.py \
        --source /path/to/source.flac \
        --target p225 \
        --output results/audio/converted.wav \
        --config configs/default.yaml
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path

from src.config import load_config
from src.audio.preprocessing import load_audio
from src.audio.features import WavLMFeatureExtractor
from src.models.knn_matcher import KNNMatcher
from src.models.vocoder import HiFiGANVocoder
from src.inference.convert import VoiceConverter


def load_speaker_features(features_dir: Path, speaker_id: str) -> torch.Tensor:
    spk_dir = features_dir / speaker_id
    if not spk_dir.exists():
        raise ValueError(f"No precomputed features for speaker '{speaker_id}'. Run extract_features.py first.")
    feats = [
        torch.from_numpy(np.load(str(p)))
        for p in sorted(spk_dir.glob("*.npy"))
    ]
    return torch.cat(feats, dim=0)  # (T, 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  required=True,  help="Path to source audio file")
    parser.add_argument("--target",  required=True,  help="Target speaker ID (e.g. p225)")
    parser.add_argument("--output",  required=True,  help="Path to save converted audio")
    parser.add_argument("--config",  default="configs/default.yaml")
    args = parser.parse_args()

    cfg          = load_config(args.config)
    features_dir = Path(cfg["data"]["features_dir"])
    sr           = cfg["audio"]["sample_rate"]
    max_dur      = cfg["audio"]["max_duration_sec"]

    # Load models
    extractor = WavLMFeatureExtractor(cfg)
    matcher   = KNNMatcher(k=cfg["knn"]["k"], use_gpu=cfg["knn"]["use_gpu"])
    vocoder   = HiFiGANVocoder(cfg)

    import time
    t_start = time.perf_counter()

    # Extract source features
    src_wav   = load_audio(args.source, sr, max_dur)
    src_feats = extractor.extract(src_wav, sr)           # (S, 1024)

    # Load precomputed target features
    tgt_feats = load_speaker_features(features_dir, args.target)  # (T, 1024)

    # kNN match + vocode
    matched   = matcher.match(src_feats, tgt_feats)      # (S, 1024)
    out_wav   = vocoder.vocode(matched)                  # (1, samples)

    t_end = time.perf_counter()

    # Save
    import torchaudio
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), out_wav, sr)

    audio_dur = src_wav.shape[-1] / sr
    rtf       = (t_end - t_start) / audio_dur

    result = {
        "source":         args.source,
        "target_speaker": args.target,
        "output":         args.output,
        "source_frames":  src_feats.shape[0],
        "target_frames":  tgt_feats.shape[0],
        "audio_duration": round(audio_dur, 3),
        "inference_time": round(t_end - t_start, 3),
        "rtf":            round(rtf, 4),
    }

    print("\n── Conversion complete ──")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
