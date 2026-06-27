import torch
import faiss
import numpy as np


class KNNMatcher:
    def __init__(self, k: int = 4, use_gpu: bool = True):
        self.k = k
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0

    def match(self, source_feats: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
        """
        source_feats : (S, D) — content features from source utterance
        target_feats : (T, D) — content features from target speaker pool
        Returns      : (S, D) — matched features (averaged over k neighbours)
        """
        S, D = source_feats.shape

        src_np = source_feats.numpy().astype(np.float32)
        tgt_np = target_feats.numpy().astype(np.float32)

        # Build FAISS index on target features
        index = faiss.IndexFlatL2(D)
        if self.use_gpu:
            res   = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(tgt_np)

        # Search: for each source frame find k nearest target frames
        _, indices = index.search(src_np, self.k)  # (S, k)

        # Average the k neighbours
        matched = tgt_np[indices].mean(axis=1)  # (S, D)

        return torch.from_numpy(matched)
