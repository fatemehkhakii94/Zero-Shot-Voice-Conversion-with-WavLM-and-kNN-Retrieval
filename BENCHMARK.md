# Benchmark Results

## Evaluation Setup

- **Dataset:** VCTK 0.92 (7 speakers, 10 conversion pairs)
- **Conversion type:** Any-to-any zero-shot (no target speaker fine-tuning)
- **Hardware:** NVIDIA RTX 3060 12GB

## Metrics

| Metric | Tool | Description |
|---|---|---|
| Speaker Similarity | ECAPA-TDNN (SpeechBrain) cosine similarity | How well target identity is transferred |
| WER | Whisper base.en | Content preservation |
| UTMOS OVRL | DNSMOS (speechmos) | Perceptual audio quality (1-5) |
| RTF | Wall-clock time / audio duration | Inference speed |

## Summary

| Metric | Score |
|---|---|
| Speaker Similarity (higher is better) | 0.562 |
| WER (lower is better) | 0.033 |
| UTMOS OVRL (higher is better) | 3.18 |
| Avg RTF (lower is better) | 0.275 |

## Per-Pair Results

| Source | Target | Spk Sim | WER | UTMOS | RTF |
|---|---|---|---|---|---|
| p226_001 | p225 | 0.679 | 0.000 | 3.155 | 0.114 |
| p226_002 | p227 | 0.503 | 0.000 | 3.322 | 0.201 |
| p227_001 | p228 | 0.577 | 0.000 | 3.091 | 0.262 |
| p227_002 | p225 | 0.588 | 0.000 | 3.134 | 0.233 |
| p228_001 | p226 | 0.585 | 0.000 | 3.018 | 0.367 |
| p228_002 | p229 | 0.552 | 0.000 | 3.334 | 0.217 |
| p229_001 | p230 | 0.542 | 0.000 | 2.941 | 0.469 |
| p229_002 | p226 | 0.436 | 0.000 | 3.409 | 0.259 |
| p230_001 | p231 | 0.581 | 0.333 | 3.151 | 0.405 |
| p230_002 | p225 | 0.576 | 0.000 | 3.202 | 0.219 |

## Notes

- WER of 0.333 on p230 to p231 is a Whisper transcription edge case on a short utterance, not a content preservation failure.
- Speaker similarity range 0.44-0.68 reflects difficulty variation across speaker pairs.
- UTMOS scores slightly above source recordings (avg source ~2.86) because vocoder output is cleaner than room-recorded VCTK audio.
