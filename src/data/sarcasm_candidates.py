"""Heuristic filter for sarcasm/irony candidate captions (text-side cues only).

Hashtag pages (even #طنز) mix plain selfies with ironic posts. When collecting
for sarcasm labels, use this to skip captions with no contrast / mismatch hints.

This does NOT detect image–text mismatch (that happens at labeling time).
"""

from __future__ import annotations

import re

from .preprocess import preprocess_caption

# Persian mood / sentiment cues (subset — high precision over recall).
_NEG_WORDS = re.compile(
    r"ناراحت|غمگین|غم|گریه|افسرد|حالم\s*بد|خسته|عصبانی|نفرت|بدبخت|"
    r" miserable|sad|depressed|tired|hate|upset",
    re.I,
)
_POS_WORDS = re.compile(
    r"خوشحال|شاد|عالی|فوق\s*العاده|love|happy|great|amazing|best\s*day",
    re.I,
)
_CONTRAST = re.compile(r"ولی|اما|در\s+حالی|although|but\s+", re.I)
_RHETORICAL = re.compile(r"ببینم|کی\s+بلد|مگه| seriously|yeah\s+right", re.I)
_IRONY_MARKERS = re.compile(r"کنایه|طنز|سرتق|iron(y|ic)|sarcasm", re.I)

# Happy emoji near sad words (or vice versa) in caption.
_HAPPY_EMOJI = re.compile(r"[\U0001F600-\U0001F64F\U00002764\U0001F970\U0001F60A\U0001F602]")
_SAD_EMOJI = re.compile(r"[\U0001F622-\U0001F62D\U0001F614\U0001F61E\U00002639]")


def is_sarcasm_candidate_caption(caption: str) -> bool:
    """True if caption text suggests irony/sarcasm (not a plain selfie caption)."""
    text = preprocess_caption(caption)
    if len(text) < 8:
        return False

    has_neg = bool(_NEG_WORDS.search(text))
    has_pos = bool(_POS_WORDS.search(text))
    has_contrast = bool(_CONTRAST.search(text))
    has_rhetorical = bool(_RHETORICAL.search(text))
    has_irony_tag = bool(_IRONY_MARKERS.search(text))
    happy_emoji = bool(_HAPPY_EMOJI.search(text))
    sad_emoji = bool(_SAD_EMOJI.search(text))

    if has_irony_tag:
        return True
    if has_neg and happy_emoji:
        return True
    if has_pos and sad_emoji:
        return True
    if has_neg and has_pos:
        return True
    if has_contrast and (has_neg or has_pos):
        return True
    if has_rhetorical and (has_neg or has_pos):
        return True
    # Quoted / meme-style captions often on image posts.
    if text.count('"') >= 2 or text.count("«") >= 1:
        return True
    return False
