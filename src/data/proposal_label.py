"""Assign the 5 proposal labels from caption polarity, image affect, and humor.

Operational definition of the thesis 5-way scheme (no sixth class):

- ``positive_sarcasm``: negative / bitter caption vs a clearly happy image
- ``negative_sarcasm``: positive / cheerful caption vs a clearly unhappy image
- ``positive`` / ``negative`` / ``neutral``: aligned or non-emotional posts

Verbal Persian jokes (``#طنز``, 😅, …) with *no* image–text polarity clash are
``positive`` (humor), not sarcasm. Ads and promotional copy are ``neutral``.
"""

from __future__ import annotations

import re

from data.preprocess import normalize_persian

# English affect in SmolVLM descriptions (not a Persian word list).
_HAT_POS = re.compile(
    r"\b(?:smil(?:e|ing|es)|laugh(?:ing|s)?|grin(?:ning)?|happy|joyful|"
    r"cheer(?:ful)?|delighted|beaming|excited)\b",
    re.I,
)
_HAT_NEG = re.compile(
    r"\b(?:sad|cry(?:ing)?|tears?|frown(?:ing)?|angry|upset|depressed|gloomy|"
    r"distressed|somber|unhappy|miserable|crying)\b",
    re.I,
)

_HUMOR = re.compile(
    r"طنز|جوک|خنده_دار|خنده‌دار|#خنده|#فان|#کمدی|😂|😅|🤣|😆",
    re.I,
)
_ADS = re.compile(
    r"ثبت[\s\-]*نام|پشتیبانی|واتساپ|تلگرام|روبیکا|برای کسب اطلاعات|"
    r"عرشیان|دورهٔ|دوره\s*«|021\d{5,}|1000\d{3,}",
    re.I,
)

# High-precision Persian cues for gold labeling when the polarity head is bland.
_IDIOM_NEG = re.compile(
    r"باور\s*نکن|حال(?:م|مون| ما)?\s*(?:اصلا\s*)?خوب\s*نیست|"
    r"حالم\s*بد|سخت\s*غمگین|شادروان|مرحوم|خدا\s*بیامرز|"
    r"زنده‌?\s*یاد|آسمانی\s*شد|انا\s*لله|یادش(?:ان)?\s*گرامی|"
    r"هر\s*هر\s*هر",
    re.I,
)
_LEX_NEG = re.compile(
    r"ناراحت|غمگین|گریه|افسرد|دلم\s*تنگ|رنج\b|بدبخت|گرون(?:ی| کردن)|"
    r"تورم|تسلیت|فوت\s+کرد|تنها بودم|بدون عشق|از دست دادن|طنز_تلخ",
    re.I,
)
_LEX_POS = re.compile(
    r"چه روز (?:زیبا|عالی)|خیلی خوشحال|دوستت دارم|مبارک|"
    r"دلتون شاد|فوق.?العاده|حال(?:م)?\s*خوبه(?!\s*نیست)",
    re.I,
)
_CELEBRATION = re.compile(r"عروسی|دوماد|تولد|مبارک|خوشبخت", re.I)
_HASHTAG = re.compile(r"#\S+")
_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)

_TEXT_COMMIT = 0.12
_CLASH_TEXT = 0.28
_HAT_COMMIT = 0.15
_CLASH_VIS_POS = 0.75
_CLASH_VIS_NEG = 0.70


def _scalar(probs) -> float:
    if probs is None or len(probs) < 2:
        return 0.0
    return float(probs[1]) - float(probs[0])


def hat_affect(description: str) -> str:
    """Return ``pos``, ``neg``, or ``neu`` from the English VLM caption."""
    t = description or ""
    pos, neg = bool(_HAT_POS.search(t)), bool(_HAT_NEG.search(t))
    if pos and not neg:
        return "pos"
    if neg and not pos:
        return "neg"
    return "neu"


def _content_letter_count(caption: str) -> int:
    cap = _HASHTAG.sub(" ", normalize_persian(caption or ""))
    return len(_LETTERS.findall(cap))


def caption_is_thin(caption: str) -> bool:
    """True when the caption is mostly hashtags / emoji, not a real sentence."""
    return _content_letter_count(caption) < 8


def text_affect(caption: str, pol_T=None) -> str:
    """Return ``pos``, ``neg``, or ``neu`` for the Persian caption."""
    cap = normalize_persian(caption or "")
    s = _scalar(pol_T)
    if _IDIOM_NEG.search(cap) or (bool(_LEX_NEG.search(cap)) and not _LEX_POS.search(cap)):
        return "neg"
    if bool(_LEX_POS.search(cap)) and not _LEX_NEG.search(cap):
        return "pos"
    if _content_letter_count(cap) < 20:
        return "neu"
    if s <= -_TEXT_COMMIT:
        return "neg"
    if s >= _TEXT_COMMIT:
        return "pos"
    return "neu"


def resolve_hat(
    *,
    generated: str = "",
    pol_T_hat=None,
    visual_hat: str | None = None,
) -> str:
    """Prefer CLIP visual affect, then VLM wording, then T̂ polarity."""
    if visual_hat in {"pos", "neg"}:
        return visual_hat
    hat = hat_affect(generated)
    if hat != "neu":
        return hat
    s_h = _scalar(pol_T_hat)
    if s_h >= _HAT_COMMIT:
        return "pos"
    if s_h <= -_HAT_COMMIT:
        return "neg"
    return "neu"


def _strong_text_neg(caption: str, pol_T) -> bool:
    cap = normalize_persian(caption or "")
    if _IDIOM_NEG.search(cap) or _LEX_NEG.search(cap):
        return True
    if _content_letter_count(cap) < 20:
        return False
    return _scalar(pol_T) <= -_CLASH_TEXT


def _strong_text_pos(caption: str, pol_T) -> bool:
    cap = normalize_persian(caption or "")
    if _IDIOM_NEG.search(cap) or _LEX_NEG.search(cap):
        return False
    if _LEX_POS.search(cap):
        return True
    if _content_letter_count(cap) < 20:
        return False
    return _scalar(pol_T) >= _CLASH_TEXT


def _strong_hat_pos(generated: str, visual_hat: str | None, visual_pos: float | None) -> bool:
    if hat_affect(generated) == "pos":
        return True
    if visual_hat != "pos":
        return False
    return visual_pos is None or visual_pos >= _CLASH_VIS_POS


def _strong_hat_neg(generated: str, visual_hat: str | None, visual_neg: float | None) -> bool:
    if hat_affect(generated) == "neg":
        return True
    if visual_hat != "neg":
        return False
    return visual_neg is None or visual_neg >= _CLASH_VIS_NEG


def assign_proposal_label(
    caption: str,
    *,
    pol_T=None,
    pol_T_hat=None,
    generated: str = "",
    visual_hat: str | None = None,
    visual_pos: float | None = None,
    visual_neg: float | None = None,
) -> str:
    """Map one (caption, image-description, polarities) triple to a spec label."""
    cap = normalize_persian(caption or "")
    text = text_affect(cap, pol_T)
    hat = resolve_hat(generated=generated, pol_T_hat=pol_T_hat, visual_hat=visual_hat)
    humor = bool(_HUMOR.search(cap))
    bitter = bool(_IDIOM_NEG.search(cap) or _LEX_NEG.search(cap))

    if _ADS.search(cap) and text != "neg":
        return "neutral"
    if caption_is_thin(cap) and not bitter and not humor:
        return "neutral"

    # Multimodal clash — only with a *clear* opposite on both sides.
    if _strong_text_neg(cap, pol_T) and _strong_hat_pos(generated, visual_hat, visual_pos):
        return "positive_sarcasm"
    if (
        _strong_text_pos(cap, pol_T)
        and _strong_hat_neg(generated, visual_hat, visual_neg)
        and not bitter
        and not _CELEBRATION.search(cap)
    ):
        return "negative_sarcasm"

    # Verbal joke / طنز with no visual clash.
    if humor:
        return "positive"

    if text == "pos":
        return "positive"
    if text == "neg":
        # Bland polarity vs a happy photo is not sarcasm and not a complaint.
        if hat == "pos" and not bitter:
            return "positive"
        return "negative"
    if hat == "pos":
        return "positive"
    if hat == "neg":
        return "negative"
    return "neutral"
