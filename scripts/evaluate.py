"""
Run full evaluation over a set of voice conversions.
Produces a JSON results file and prints a summary table.

Usage:
    PYTHONPATH=. python scripts/evaluate.py \
        --config configs/default.yaml \
        --pairs eval_pairs.txt \
        --output results/metrics/eval_results.json
"""
import argparse
import json
import time
import torchaudio
from pathlib import Path

from src.config import load_config
from src.audio.preprocessing import load_audio
from src.audio.features import WavLMFeatureExtractor
from src.models.knn_matcher import KNNMatcher
from src.models.vocoder import HiFiGANVocoder
from src.evaluation.speaker_similarity import SpeakerSimilarity
from src.evaluation.wer import WERScorer
from src.evaluation.utmos import UTMOSScorer

import numpy as np
import torch


def load_speaker_features(features_dir: Path, speaker_id: str) -> torch.Tensor:
    spk_dir = features_dir / speaker_id
    feats   = [
        torch.from_numpy(np.load(str(p)))
        for p in sorted(spk_dir.glob("*.npy"))
    ]
    return torch.cat(feats, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pairs",  required=True,
                        help="Text file: each line = source_path target_speaker_id")
    parser.add_argument("--output", default="results/metrics/eval_results.json")
    args = parser.parse_args()

    cfg          = load_config(args.config)
    features_dir = Path(cfg["data"]["features_dir"])
    audio_dir    = Path(cfg["results"]["audio_dir"])
    sr           = cfg["audio"]["sample_rate"]
    max_dur      = cfg["audio"]["max_duration_sec"]

    # Load models
    extractor  = WavLMFeatureExtractor(cfg)
    matcher    = KNNMatcher(k=cfg["knn"]["k"], use_gpu=cfg["knn"]["use_gpu"])
    vocoder    = HiFiGANVocoder(cfg)
    sim_model  = SpeakerSimilarity(device=cfg["wavlm"]["device"])
    wer_scorer = WERScorer(model_size="base.en", device=cfg["wavlm"]["device"])
    mos_scorer = UTMOSScorer(device=cfg["wavlm"]["device"])

    # Parse pairs file
    pairs = []
    with open(args.pairs) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            src, tgt = line.split()
            pairs.append((src, tgt))

    print(f"Evaluating {len(pairs)} pairs ...")
    results = []

    for src_path, tgt_id in pairs:
        src_path = Path(src_path)
        out_path = audio_dir / f"{src_path.stem}_to_{tgt_id}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert
        t0      = time.perf_counter()
        src_wav = load_audio(src_path, sr, max_dur)
        src_f   = extractor.extract(src_wav, sr)
        tgt_f   = load_speaker_features(features_dir, tgt_id)
        matched = matcher.match(src_f, tgt_f)
        out_wav = vocoder.vocode(matched)
        t1      = time.perf_counter()

        torchaudio.save(str(out_path), out_wav, sr)

        audio_dur = src_wav.shape[-1] / sr
        rtf       = (t1 - t0) / audio_dur

        # Target reference: first utterance of target speaker
        tgt_ref = sorted(
            (Path(cfg["data"]["vctk_root"]) / "wav48_silence_trimmed" / tgt_id).glob("*.flac")
        )[0]
        tgt_ref_wav = load_audio(tgt_ref, sr, max_dur)

        # Metrics
        sim    = sim_model.score(out_wav, tgt_ref_wav)
        wer    = wer_scorer.score(src_path, out_path)
        mos    = mos_scorer.score(out_path)

        row = {
            "source":       str(src_path),
            "target":       tgt_id,
            "output":       str(out_path),
            "speaker_sim":  round(sim, 4),
            "wer":          wer["wer"],
            "reference":    wer["reference"],
            "hypothesis":   wer["hypothesis"],
            "utmos_ovrl":   mos["ovrl"],
            "rtf":          round(rtf, 4),
        }
        results.append(row)

        print(f"  {src_path.name} → {tgt_id} | sim={sim:.3f} wer={wer['wer']:.3f} mos={mos['ovrl']:.3f} rtf={rtf:.3f}")

    # Aggregate
    avg = {
        "avg_speaker_sim": round(sum(r["speaker_sim"] for r in results) / len(results), 4),
        "avg_wer":         round(sum(r["wer"]         for r in results) / len(results), 4),
        "avg_utmos_ovrl":  round(sum(r["utmos_ovrl"]  for r in results) / len(results), 4),
        "avg_rtf":         round(sum(r["rtf"]          for r in results) / len(results), 4),
    }

    output = {"summary": avg, "pairs": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print("\n── Summary ──")
    for k, v in avg.items():
        print(f"  {k}: {v}")
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
