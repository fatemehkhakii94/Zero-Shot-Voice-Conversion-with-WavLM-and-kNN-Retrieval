"""
Extract WavLM features for all VCTK speakers and save to disk.
Usage: python scripts/extract_features.py --config configs/default.yaml
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.config import load_config
from src.audio.preprocessing import load_audio
from src.audio.features import WavLMFeatureExtractor


def get_speakers(vctk_root: Path, max_speakers: int | None) -> list[Path]:
    speaker_dirs = sorted([
        d for d in (vctk_root / "wav48_silence_trimmed").iterdir()
        if d.is_dir() and d.name != "log.txt"
    ])
    if max_speakers:
        speaker_dirs = speaker_dirs[:max_speakers]
    return speaker_dirs


def main(cfg: dict):
    vctk_root    = Path(cfg["data"]["vctk_root"])
    features_dir = Path(cfg["data"]["features_dir"])
    ext          = cfg["data"]["audio_ext"]
    sr           = cfg["audio"]["sample_rate"]
    max_dur      = cfg["audio"]["max_duration_sec"]
    max_spk      = cfg["data"]["max_speakers"]
    speakers     = cfg["data"]["speakers"]

    features_dir.mkdir(parents=True, exist_ok=True)
    extractor = WavLMFeatureExtractor(cfg)

    if speakers:
        speaker_dirs = [vctk_root / "wav48_silence_trimmed" / s for s in speakers]
    else:
        speaker_dirs = get_speakers(vctk_root, max_spk)

    print(f"Processing {len(speaker_dirs)} speakers → {features_dir}")

    for spk_dir in tqdm(speaker_dirs, desc="Speakers"):
        spk_id  = spk_dir.name
        out_dir = features_dir / spk_id
        out_dir.mkdir(parents=True, exist_ok=True)

        audio_files = sorted(spk_dir.glob(f"*.{ext}"))
        if not audio_files:
            print(f"  WARNING: no .{ext} files found for {spk_id}")
            continue

        for audio_path in tqdm(audio_files, desc=spk_id, leave=False):
            out_path = out_dir / (audio_path.stem + ".npy")
            if out_path.exists():
                continue  # resume-friendly

            wav   = load_audio(audio_path, sr, max_dur)
            feats = extractor.extract(wav, sr)           # (frames, 1024)
            np.save(str(out_path), feats.numpy())

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args   = parser.parse_args()
    cfg    = load_config(args.config)
    main(cfg)
