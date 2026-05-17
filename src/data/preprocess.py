"""Persian caption preprocessing.

Pure stdlib so it stays unit-testable without heavy dependencies. The semantics
match the spec's `Persian Text Preprocessing` requirement:

- Normalize Arabic forms to Persian (`ي → ی`, `ك → ک`).
- Unify Arabic-Indic and Persian-Indic digits to ASCII.
- Strip URLs and `@mentions`.
- Collapse whitespace.
- Filter captions that are <50 percent Persian by character count.
"""

from __future__ import annotations

import re
import unicodedata

# Mapping for character normalization. The right-hand side is the canonical
# Persian form.
_CHAR_MAP: dict[str, str] = {
    "ي": "ی",  # Arabic Yeh
    "ك": "ک",  # Arabic Kaf
    "ى": "ی",  # Alef Maksura
    "ٱ": "ا",
    "أ": "ا",
    "إ": "ا",
    "آ": "آ",
}

# Arabic-Indic digits (U+0660–0669) and Persian-Indic digits (U+06F0–06F9)
# mapped to ASCII 0-9.
_DIGIT_MAP: dict[str, str] = {
    **{chr(0x0660 + i): str(i) for i in range(10)},
    **{chr(0x06F0 + i): str(i) for i in range(10)},
}

# Combined translation table for cheap O(1) replacement.
_TRANSLATION_TABLE = {ord(k): v for k, v in {**_CHAR_MAP, **_DIGIT_MAP}.items()}

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<![\w])@\w+")
_WHITESPACE_RE = re.compile(r"\s+")

# Persian letter range: U+0600 — U+06FF (Arabic block, includes Persian).
# We exclude digits since they were already normalized to ASCII above.
_PERSIAN_LETTER_RE = re.compile(r"[\u0600-\u06FF]")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def normalize_persian(text: str) -> str:
    """Apply Unicode NFC then map Arabic forms and digits to Persian/ASCII."""
    if not text:
        return ""
    nfc = unicodedata.normalize("NFC", text)
    return nfc.translate(_TRANSLATION_TABLE)


def strip_noise(text: str) -> str:
    """Remove URLs, mentions, and collapse whitespace."""
    out = _URL_RE.sub(" ", text)
    out = _MENTION_RE.sub(" ", out)
    out = _WHITESPACE_RE.sub(" ", out).strip()
    return out


def preprocess_caption(text: str) -> str:
    """Full pipeline: normalize -> strip noise."""
    return strip_noise(normalize_persian(text))


def persian_ratio(text: str) -> float:
    """Return the fraction of letter characters that fall in the Persian block.

    Non-letter characters (digits, punctuation, emoji, whitespace) are ignored
    so a caption like "hello! 😊" returns 0.0 and "سلام دنیا" returns 1.0.
    """
    letters = _LETTER_RE.findall(text)
    if not letters:
        return 0.0
    persian = sum(1 for ch in letters if _PERSIAN_LETTER_RE.match(ch))
    return persian / len(letters)


def is_persian_enough(text: str, *, min_ratio: float = 0.5) -> bool:
    """Spec scenario "Language filter": at least 50 percent Persian letters."""
    return persian_ratio(text) >= min_ratio
