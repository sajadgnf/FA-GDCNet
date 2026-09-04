"""Heuristic filter for sarcasm/irony candidate captions (text-side cues only).

Hashtag pages (even #طنز / #کنایه) are mostly news, quote-cards, and verbal
jokes. Collection must require a *clash cue in the caption*, not the tag name.

This does NOT detect image–text mismatch (that happens at labeling time).
"""

from __future__ import annotations

import re

from .preprocess import preprocess_caption

# Persian mood / sentiment cues (subset — high precision over recall).
_NEG_WORDS = re.compile(
    r"ناراحت|غمگین|غم|گریه|افسرد|حالم\s*بد|خسته|عصبانی|نفرت|بدبخت|"
    r"تلخ|تنها(?:م|یی)?|دپرس|دلشکسته|اشک|"
    r" miserable|sad|depressed|tired|hate|upset",
    re.I,
)
_POS_WORDS = re.compile(
    r"خوشحال|شاد|عالی|فوق\s*العاده|لبخند|میخندم|love|happy|great|amazing|best\s*day",
    re.I,
)
_CONTRAST = re.compile(r"ولی|اما|در\s+حالی|although|but\s+", re.I)
_RHETORICAL = re.compile(r"ببینم|کی\s+بلد|مگه| seriously|yeah\s+right", re.I)
_BITTER = re.compile(r"باور\s*نکن|طنز_تلخ|تلخند|هر\s*هر\s*هر", re.I)
_HASHTAG = re.compile(r"#\S+")
_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)
_NEWS_META = re.compile(
    r"وایرال\s*شد|واکنش\s*کنایه|استوری\s*کنایه|توئیت\s*کنایه|"
    r"با\s*لحن(?:ی)?\s*کنایه|خبرفوری|#خبر\b|"
    r"پس\s+از\s+پخش|شایعات\s+مربوط|رسانه\s*های\s+اجتماعی|"
    r"کنایه\s*آمیز\s+(?:به|خطاب)|خطاب\s+به|"
    r"عضو\s+مجلس|نشست\s+خبری",
    re.I,
)
# Happy emoji near sad words (or vice versa) in caption.
_HAPPY_EMOJI = re.compile(r"[\U0001F600-\U0001F64F\U00002764\U0001F970\U0001F60A\U0001F602]")
_SAD_EMOJI = re.compile(r"[\U0001F622-\U0001F62D\U0001F614\U0001F61E\U00002639]")

# Political / Iran-state satire. Do not match bare «ایران» / «ایرانی»
# (explore tags, «طنز_ایرانی», «دابسمش_ایرانی»).
_POLITICAL_FA = (
    "خامنه",
    "خمینی",
    "رئیسی",
    "رییسی",
    "روحانی",
    "احمدی‌نژاد",
    "احمدینژاد",
    "احمدی نژاد",
    "پزشکیان",
    "قالیباف",
    "جلیلی",
    "ولایت فقیه",
    "ولی فقیه",
    "ولی‌فقیه",
    "مقام معظم",
    "رهبر انقلاب",
    "جمهوری اسلامی",
    "انقلاب 57",
    "انقلاب ۵۷",
    "انقلاب اسلامی",
    "طنزسیاسی",
    "طنز_سیاسی",
    "طنز سیاسی",
    "سیاسی",
    "مسئولین",
    "رئیس جمهور",
    "رئیس‌جمهور",
    "ریاست جمهوری",
    "صدا و سیما",
    "صداوسیما",
    "صدا_و_سیما",
    "سپاه",
    "بسیج",
    "برجام",
    "تحریم",
    "انتخابات",
    "اصولگرا",
    "اصلاح طلب",
    "اصلاح‌طلب",
    "زن زندگی آزادی",
    "زن_زندگی_آزادی",
    "مهسا امینی",
    "مهسا_امینی",
    "ژینا",
    "استبداد",
    "قطعی برق",
    "قطعی_برق",
    "اسرائیل",
    "فلسطین",
    "حماس",
    "حزب الله",
    "حزب‌الله",
    "غزه",
    "#تورم",
    "#گرانی",
    "#دولت",
    "وزیر",
    "نماینده مجلس",
)
_POLITICAL_LATIN = (
    "khamenei",
    "khomeini",
    "raisi",
    "irgc",
    "islamic republic",
    "zionist",
    "netanyahu",
    "#iranprotest",
)


def body_letter_count(caption: str) -> int:
    """Letters left after stripping hashtags (quote-cards score near zero)."""
    text = _HASHTAG.sub(" ", preprocess_caption(caption))
    return len(_LETTERS.findall(text))


def is_news_or_meta_caption(caption: str) -> bool:
    """True for gossip/news *about* someone else's sarcastic story."""
    return bool(_NEWS_META.search(preprocess_caption(caption)))


def is_sarcasm_candidate_caption(caption: str, *, allow_weak_cues: bool = True) -> bool:
    """True if the caption itself has a polarity-clash cue.

    Irony hashtags (``#کنایه``, ``#طنز``, ``#تیکه``) are **not** enough — those
    pages are news screenshots and quote-cards. ``allow_weak_cues=False`` also
    drops contrast-only / rhetorical-only matches.
    """
    text = preprocess_caption(caption)
    if body_letter_count(caption) < 12:
        return False
    if is_news_or_meta_caption(text):
        return False

    has_neg = bool(_NEG_WORDS.search(text))
    has_pos = bool(_POS_WORDS.search(text))
    has_contrast = bool(_CONTRAST.search(text))
    has_rhetorical = bool(_RHETORICAL.search(text))
    happy_emoji = bool(_HAPPY_EMOJI.search(text))
    sad_emoji = bool(_SAD_EMOJI.search(text))

    if bool(_BITTER.search(text)):
        return True
    if has_neg and happy_emoji:
        return True
    if has_pos and sad_emoji:
        return True
    if has_neg and has_pos:
        return True
    if has_contrast and (has_neg or has_pos):
        return True
    if not allow_weak_cues:
        return False
    if has_rhetorical and (has_neg or has_pos):
        return True
    return False


def is_political_caption(caption: str) -> bool:
    """True for political / Iran-state captions that must not enter the pool.

    Bare «ایران» or «ایرانی» is not enough (explore tags, comedy pages).
    """
    text = preprocess_caption(caption)
    if not text:
        return False
    lower = text.lower()
    if any(s in lower for s in _POLITICAL_LATIN):
        return True
    return any(s in text for s in _POLITICAL_FA)
