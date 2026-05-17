"""Tests for the Persian text preprocessor (pure stdlib)."""

from __future__ import annotations

from data.preprocess import (
    is_persian_enough,
    normalize_persian,
    persian_ratio,
    preprocess_caption,
    strip_noise,
)


def test_normalize_arabic_to_persian():
    raw = "كتاب علي ي"
    out = normalize_persian(raw)
    assert "ك" not in out and "ي" not in out
    assert "ک" in out and "ی" in out


def test_normalize_digits():
    raw = "تماس ۰۹۱۲۱۲۳۴۵۶۷ یا ٠٩١٢١٢٣٤٥٦٧"
    out = normalize_persian(raw)
    assert "09121234567" in out
    assert "۰" not in out and "٠" not in out


def test_strip_urls_and_mentions():
    raw = "ببین این لینک رو https://example.com/foo و @ali چه میگه"
    out = strip_noise(raw)
    assert "https://example.com/foo" not in out
    assert "@ali" not in out
    assert "ببین" in out


def test_collapses_whitespace():
    raw = "سلام   دنیا\n\nخوبی"
    out = strip_noise(raw)
    assert "  " not in out
    assert "\n" not in out


def test_preprocess_caption_pipeline():
    raw = "كتاب @ali  https://x.com/ai رو ببين ۲۳ ساله"
    out = preprocess_caption(raw)
    assert "@ali" not in out
    assert "https" not in out
    assert "ک" in out and "ی" in out
    assert "23" in out


def test_persian_ratio_pure_persian():
    assert persian_ratio("سلام دنیا") == 1.0


def test_persian_ratio_mixed():
    # 4 Persian letters + 5 English letters → 4/9
    ratio = persian_ratio("سلام hello")
    assert 0.3 < ratio < 0.5


def test_persian_ratio_no_letters():
    assert persian_ratio("!!! 123 ?") == 0.0


def test_is_persian_enough_threshold():
    assert is_persian_enough("سلام دنیا") is True
    assert is_persian_enough("hello world") is False
    assert is_persian_enough("سلام hi how are you doing today my friend") is False


def test_normalize_preserves_arabic_diacritics_position():
    raw = "آرام"
    out = normalize_persian(raw)
    assert out.startswith("آ")
