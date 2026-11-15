from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from knnvc.config import MatchingConfig, resolve_device


@dataclass(frozen=True)
class MatchSummary:
    query_frames: int
    reference_frames: int
    feature_dimension: int
    k: int
    metric: str


class KNNMatcher:
    """Memory-bounded exact k-nearest-neighbor matching."""

    def __init__(self, config: MatchingConfig):
        self.config = config
        self.device = resolve_device(config.device)

    @torch.inference_mode()
    def match(
        self,
        query: torch.Tensor,
        references: torch.Tensor,
    ) -> tuple[torch.Tensor, MatchSummary]:
        query = _validate_features("query", query)
        references = _validate_features("references", references)
        if query.shape[1] != references.shape[1]:
            raise ValueError(
                "Query and reference feature dimensions differ: "
                f"{query.shape[1]} != {references.shape[1]}"
            )
        if self.config.k > references.shape[0]:
            raise ValueError(
                f"matching.k={self.config.k} exceeds the {references.shape[0]} reference frames"
            )

        matched_chunks: list[torch.Tensor] = []
        for query_start in range(0, query.shape[0], self.config.query_chunk_size):
            query_chunk = query[query_start : query_start + self.config.query_chunk_size].to(
                self.device
            )
            best_values: torch.Tensor | None = None
            best_indices: torch.Tensor | None = None

            for reference_start in range(0, references.shape[0], self.config.reference_chunk_size):
                reference_chunk = references[
                    reference_start : reference_start + self.config.reference_chunk_size
                ].to(self.device)
                values = self._pairwise_values(query_chunk, reference_chunk)
                local_k = min(self.config.k, reference_chunk.shape[0])
                chunk_values, chunk_indices = torch.topk(
                    values,
                    k=local_k,
                    dim=1,
                    largest=self.config.metric == "cosine",
                )
                chunk_indices = chunk_indices + reference_start

                if best_values is None:
                    best_values = chunk_values
                    best_indices = chunk_indices
                    continue

                assert best_indices is not None
                merged_values = torch.cat((best_values, chunk_values), dim=1)
                merged_indices = torch.cat((best_indices, chunk_indices), dim=1)
                keep = min(self.config.k, merged_values.shape[1])
                best_values, selected = torch.topk(
                    merged_values,
                    k=keep,
                    dim=1,
                    largest=self.config.metric == "cosine",
                )
                best_indices = torch.gather(merged_indices, dim=1, index=selected)

            if best_indices is None or best_indices.shape[1] != self.config.k:
                raise RuntimeError("Unable to retrieve the configured number of neighbors")
            neighbor_features = references[best_indices.cpu()]
            matched_chunks.append(neighbor_features.mean(dim=1))

        matched = torch.cat(matched_chunks, dim=0).to(torch.float32)
        summary = MatchSummary(
            query_frames=query.shape[0],
            reference_frames=references.shape[0],
            feature_dimension=query.shape[1],
            k=self.config.k,
            metric=self.config.metric,
        )
        return matched, summary

    def _pairwise_values(self, query: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        if self.config.metric == "cosine":
            normalized_query = functional.normalize(query, p=2, dim=1)
            normalized_references = functional.normalize(references, p=2, dim=1)
            return normalized_query @ normalized_references.transpose(0, 1)
        return torch.cdist(query, references, p=2)


def _validate_features(name: str, features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"{name} features must be a non-empty 2D tensor")
    if not features.is_floating_point():
        raise TypeError(f"{name} features must use a floating-point dtype")
    if not torch.isfinite(features).all():
        raise ValueError(f"{name} features contain NaN or infinite values")
    return features.detach().to(device="cpu", dtype=torch.float32).contiguous()
