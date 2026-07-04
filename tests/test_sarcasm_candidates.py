"""Tests for sarcasm caption heuristics."""

from data.sarcasm_candidates import is_sarcasm_candidate_caption


def test_sad_text_happy_emoji() -> None:
    assert is_sarcasm_candidate_caption("من خیلی ناراحتم 😂😊")


def test_contrast_clause() -> None:
    assert is_sarcasm_candidate_caption("حالم بده ولی دارم میخندم")


def test_plain_selfie_not_candidate() -> None:
    assert not is_sarcasm_candidate_caption("سلفی امروز #سلفی #روزمره")


def test_irony_hashtag_in_caption() -> None:
    assert is_sarcasm_candidate_caption("یکمی طنز برای شب #طنز")
