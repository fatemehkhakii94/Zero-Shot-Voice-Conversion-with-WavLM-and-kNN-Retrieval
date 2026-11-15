# VCTK 0.92 benchmark

This benchmark evaluates target-speaker transfer, content preservation, predicted speech quality, and inference speed on a deterministic subset of VCTK 0.92. The complete pair-level evidence is in [`report.json`](report.json); the split is in [`../../data/manifests/vctk-0.92-evaluation.yaml`](../../data/manifests/vctk-0.92-evaluation.yaml).

## Protocol

| Property | Value |
|---|---|
| Dataset | VCTK 0.92, microphone 1, silence-trimmed audio |
| Speakers | p225, p226, p234, p237, p238, p243, p244, p245 |
| Cohort | 4 female, 4 male; English, Scottish, Northern Irish, and Irish accents |
| Pairing | Cyclic cross-gender pairing; 5 source utterances per speaker |
| Evaluation pairs | 40 |
| Target bank | 20 utterances per target speaker |
| Held-out target references | 3 utterances per target speaker |
| Retrieval | Exact cosine kNN, k=4, WavLM-Large layer 6 |
| Synthesis | Released prematched HiFi-GAN, 16 kHz |

For each speaker, files are sorted lexically and partitioned into five source utterances, three held-out target references, and twenty bank references. These sets are disjoint. A target reference is never used to build the bank it evaluates.

The pair direction is:

```text
p225 → p226 → p234 → p237 → p238 → p243 → p244 → p245 → p225
```

## Results

| Metric | Result | 95% bootstrap CI |
|---|---:|---:|
| Converted-to-target ECAPA cosine | 0.6407 | [0.6140, 0.6651] |
| Source-to-target ECAPA cosine | 0.0820 | [0.0588, 0.1025] |
| Target-similarity gain | **+0.5587** | [0.5324, 0.5835] |
| Converted corpus WER | 9.06% | — |
| Source corpus WER | 4.17% | — |
| Converted DNSMOS overall | 3.2501 | [3.2045, 3.2932] |
| Source DNSMOS overall | 3.2646 | [3.2150, 3.3137] |
| Median inference RTF | 0.0179 | — |
| Mean / p95 inference RTF | 0.0208 / 0.0348 | — |

The primary success signal is the large positive target-similarity gain. The trade-off is a 4.89 percentage-point increase in corpus WER. Predicted quality is effectively flat on this sample (-0.015 DNSMOS overall). Timing is model inference only, not model loading, file I/O, or metric computation.

## Metrics

- **Content:** OpenAI Whisper `base.en`, scored against VCTK ground-truth transcripts. Corpus WER is total edit errors divided by total reference words; source WER is retained as an ASR difficulty baseline.
- **Speaker transfer:** SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`). Each converted/source embedding is averaged against three held-out target recordings.
- **Quality:** Microsoft DNSMOS P.835 through TorchMetrics, reported for source and converted audio.
- **Timing:** one warm-up followed by three measured conversions per pair. The report stores every measured duration and summarizes per-pair median real-time factors.
- **Uncertainty:** deterministic, 2,000-sample bootstrap confidence intervals with seed 0 for mean pair-level metrics.

## Environment and provenance

The checked run used Python 3.11.18, PyTorch 2.6.0+cu124, and an NVIDIA GeForce RTX 2070 Super.

| Artifact | SHA-256 |
|---|---|
| WavLM-Large checkpoint | `6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f` |
| Prematched HiFi-GAN checkpoint | `f924c7632c6eaf99004386d62293e124419e33582573552a6ae976eb88ed2dd5` |
| Evaluation manifest | `b3123bf7a20e3387bf4de21842601dea86d74fa59da6ecb5b76338750f5ea2fd` |

Feature banks record the hashes above plus every reference filename, reference SHA-256, duration, feature shape, and feature-file SHA-256. A conversion refuses an incompatible bank.

## Reproduction

Download [VCTK 0.92](https://doi.org/10.7488/ds/2645), then run:

```bash
knnvc bank build-manifest \
  --manifest data/manifests/vctk-0.92-evaluation.yaml \
  --dataset-root /path/to/VCTK-Corpus-0.92

knnvc evaluate \
  --manifest data/manifests/vctk-0.92-evaluation.yaml \
  --dataset-root /path/to/VCTK-Corpus-0.92 \
  --output benchmarks/vctk-0.92/report.json \
  --overwrite
```

The command regenerates one WAV per pair under `artifacts/audio/evaluation/vctk-0.92/`. Generated WAVs and feature banks are ignored; the report is the durable evidence artifact.

## Scope

This is a compact implementation benchmark, not a leaderboard comparison or human perceptual study. The cohort is balanced and accent-diverse but too small to support broad fairness or generalization claims.
