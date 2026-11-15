from knnvc.evaluation.metrics import normalize_transcript, word_error_score


def test_transcript_normalization() -> None:
    assert normalize_transcript("  Please, CALL Stella!  ") == "please call stella"


def test_word_error_score_uses_ground_truth_words() -> None:
    score = word_error_score("ask her to bring the bag", "ask her bring a bag")
    assert score.errors == 2
    assert score.reference_words == 6
    assert score.wer == 2 / 6


def test_empty_reference_is_well_defined() -> None:
    assert word_error_score("", "").wer == 0.0
    assert word_error_score("", "unexpected").wer == 1.0
