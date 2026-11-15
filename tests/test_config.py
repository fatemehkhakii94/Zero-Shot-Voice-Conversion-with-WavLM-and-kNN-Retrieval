from pathlib import Path

import pytest

from knnvc.config import load_config


def test_default_config_is_valid() -> None:
    config = load_config("configs/default.yaml")
    assert config.audio.sample_rate == 16_000
    assert config.model.feature_layer == 6
    assert config.matching.metric == "cosine"


@pytest.mark.parametrize(
    "contents, message",
    [
        ("model:\n  feature_layer: 12\n", "feature_layer must be 6"),
        ("matching:\n  metric: dot\n", "metric must be 'cosine' or 'l2'"),
        ("model:\n  mystery: true\n", "Unknown configuration key"),
        ("unknown:\n  value: true\n", "Unknown configuration section"),
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, contents: str, message: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)
