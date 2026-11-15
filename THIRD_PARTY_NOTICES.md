# Third-party notices

This repository contains original project code under the MIT License and interoperates with third-party models, software, and data under their own terms.

## Runtime model assets

Model weights are downloaded at runtime to user caches and are not redistributed in this repository.

- [`bshall/knn-vc`](https://github.com/bshall/knn-vc): released kNN-VC loader, prematched HiFi-GAN checkpoint, and WavLM checkpoint integration.
- [Microsoft WavLM](https://github.com/microsoft/unilm/tree/master/wavlm): self-supervised speech representation model.
- [OpenAI Whisper](https://github.com/openai/whisper): transcription model used for benchmark WER.
- [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb): speaker embeddings used for cosine similarity.
- [Microsoft DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS): non-intrusive speech-quality models accessed through TorchMetrics.

Consult each linked project for its license, model terms, and citation requirements.

## VCTK Corpus 0.92

The compact listening example contains two VCTK excerpts and one converted derivative:

> Yamagishi, Junichi; Veaux, Christophe; MacDonald, Kirsten. (2019). *CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92).* University of Edinburgh, CSTR. <https://doi.org/10.7488/ds/2645>

VCTK 0.92 is distributed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). The excerpts were resampled to mono 16 kHz PCM; the converted derivative additionally underwent WavLM feature extraction, kNN retrieval, and neural vocoding.
