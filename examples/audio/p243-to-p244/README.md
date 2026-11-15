# Listening example: p243 → p244

This example converts VCTK speaker p243 (male, London English) to p244 (female, Manchester English). It was selected from the fixed benchmark because it preserved all eleven source words while producing the largest target-similarity gain among zero-WER pairs.

1. [Source — p243](source.wav): “Ask her to bring these things with her from the store.”
2. [Target reference — p244](target-reference.wav): “When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.”
3. [Converted — p243 content in the p244 target voice](converted.wav)

| Metric | Value |
|---|---:|
| Source → target ECAPA cosine | 0.0525 |
| Converted → target ECAPA cosine | 0.7321 |
| Target-similarity gain | **+0.6797** |
| Converted WER | **0.0%** |
| Converted DNSMOS overall | 3.4722 |
| Median inference time | 86 ms |

All three files are mono PCM WAV at 16 kHz. The target reference is held out from p244's feature bank.

The source and target-reference excerpts come from [VCTK Corpus 0.92](https://doi.org/10.7488/ds/2645), © University of Edinburgh/CSTR contributors, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The converted file is derived from those materials. Changes include resampling, mono conversion, feature retrieval, and neural vocoding.
