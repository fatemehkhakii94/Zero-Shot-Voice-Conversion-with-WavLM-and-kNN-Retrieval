import pytest
import torch

from knnvc.config import MatchingConfig
from knnvc.matching import KNNMatcher


def test_cosine_matching_averages_nearest_neighbors() -> None:
    matcher = KNNMatcher(
        MatchingConfig(
            k=2,
            metric="cosine",
            device="cpu",
            query_chunk_size=1,
            reference_chunk_size=2,
        )
    )
    query = torch.tensor([[0.9, 0.1]])
    references = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    matched, summary = matcher.match(query, references)

    assert torch.allclose(matched, torch.tensor([[0.5, 0.5]]))
    assert summary.k == 2
    assert summary.metric == "cosine"


def test_chunked_l2_matches_expected_reference() -> None:
    matcher = KNNMatcher(MatchingConfig(k=1, metric="l2", device="cpu", reference_chunk_size=1))
    query = torch.tensor([[1.9, 2.1], [-0.9, -1.1]])
    references = torch.tensor([[2.0, 2.0], [-1.0, -1.0], [9.0, 9.0]])
    matched, _ = matcher.match(query, references)
    assert torch.allclose(matched, references[:2])


def test_matching_rejects_invalid_k() -> None:
    matcher = KNNMatcher(MatchingConfig(k=3, device="cpu"))
    with pytest.raises(ValueError, match="exceeds"):
        matcher.match(torch.ones(1, 2), torch.ones(2, 2))
