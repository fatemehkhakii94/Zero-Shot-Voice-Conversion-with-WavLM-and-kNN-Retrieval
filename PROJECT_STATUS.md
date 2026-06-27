# Project Status

## Status: Phase 2 Complete ✅

## Completed

- [x] Repo scaffold and config system
- [x] Audio preprocessing (load, resample, mono, clip)
- [x] WavLM-Large feature extraction (authors' checkpoint via torch.hub)
- [x] FAISS kNN matcher (GPU-accelerated)
- [x] HiFi-GAN vocoder (authors' pretrained checkpoint)
- [x] End-to-end conversion pipeline
- [x] Feature extraction for 30 VCTK speakers (~3 min)
- [x] Speaker similarity evaluation (ECAPA-TDNN)
- [x] WER evaluation (Whisper base.en)
- [x] UTMOS quality evaluation (DNSMOS)
- [x] Batch evaluation script with JSON output
- [x] README, BENCHMARK, PROJECT_STATUS

## Key Decisions

| Decision | Reason |
|---|---|
| Use authors' WavLM checkpoint (torch.hub) not HuggingFace | HiFi-GAN was trained on authors' features — mismatched features hurt quality |
| mic1 only (VCTK) | Standard choice in literature; mic2 adds no value |
| DNSMOS instead of UTMOS pip package | UTMOS depends on fairseq which fails to build on Python 3.10+ |
| Precompute and cache features | Avoids re-extracting 315M param model on every conversion |
| RTF measured end-to-end | More honest than measuring vocoder-only speed |

## Next Steps (Optional — Phase 3)

- [ ] HuBERT-Large vs WavLM-Large ablation (same pipeline, swap encoder)
- [ ] Expand evaluation to full VCTK (109 speakers)
- [ ] t-SNE visualization of speaker embeddings before/after conversion
- [ ] Spectrogram visualizations for README

## Environment

- Python 3.10
- PyTorch 2.6.0+cu124
- FAISS-GPU 1.7.4
- NVIDIA RTX 3060 12GB / CUDA 12.4
