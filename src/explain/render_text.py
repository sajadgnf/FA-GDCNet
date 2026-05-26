"""HTML heatmap renderer for Persian token-level attention.

The spec requires:
- One `<span>` per Persian token with background opacity proportional to its
  RTL-remapped attention weight.
- A fragment that displays correctly in RTL.

We emit a self-contained snippet wrapped in `<div dir="rtl">` so it renders
properly when embedded in the Streamlit dashboard or saved to disk.
"""

from __future__ import annotations

import html
from typing import Sequence

from .rtl import is_anchor, remap_rtl_indices

# WordPiece / SentencePiece prefixes used by typical Persian tokenizers.
_PIECE_PREFIXES = ("##",)
_SP_PREFIX = "▁"  # SentencePiece
# Attach to the previous token in RTL heatmaps (punctuation / affixes).
_ATTACH_CHARS = frozenset("!.,;:?؟،٪٫…")
_RLM = "\u200f"


def _is_continuation_token(token: str) -> bool:
    return token.startswith(_PIECE_PREFIXES)


def _is_attach_to_previous(display: str) -> bool:
    return bool(display) and all(ch in _ATTACH_CHARS for ch in display)


def _clean_token(t: str) -> str:
    """Strip tokenizer artifacts for display while preserving the word boundary."""
    if t.startswith(_SP_PREFIX):
        return t[len(_SP_PREFIX) :] or " "
    for p in _PIECE_PREFIXES:
        if t.startswith(p):
            return t[len(p) :]
    return t


def _normalize_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    span = hi - lo
    if span <= 0:
        return [0.0 for _ in scores]
    return [(s - lo) / span for s in scores]


def _merge_tokens_for_display(
    tokens: Sequence[str],
    scores: Sequence[float],
) -> tuple[list[str], list[float]]:
    """Merge WordPiece fragments and trailing punctuation for natural RTL lines."""
    merged_text: list[str] = []
    merged_scores: list[float] = []
    for tok, score in zip(tokens, scores):
        piece = _clean_token(tok)
        if not piece.strip():
            continue
        if merged_text and (_is_continuation_token(tok) or _is_attach_to_previous(piece)):
            merged_text[-1] += piece
            merged_scores[-1] = max(merged_scores[-1], float(score))
        else:
            merged_text.append(piece)
            merged_scores.append(float(score))
    return merged_text, merged_scores


_RTL_BLOCK_STYLE = (
    "direction:rtl;text-align:right;unicode-bidi:plaintext;"
    "font-family:Tahoma,'Segoe UI',system-ui,sans-serif;line-height:2;"
)


def render_text_heatmap(
    tokens: Sequence[str],
    scores: Sequence[float],
    *,
    title: str | None = None,
    rtl: bool = True,
    remap_tokens: bool | None = None,
    hide_anchors: bool = True,
) -> str:
    """Return an HTML fragment with per-token attention highlighting.

    When ``rtl=True`` (default), the browser lays out tokens right-to-left and
    tokenizer order is kept as-is. Set ``remap_tokens=True`` only for LTR-only
    viewers that need the legacy index reversal from ``remap_rtl_indices``.
    """
    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must have the same length")
    toks = list(tokens)
    scrs = [float(s) for s in scores]
    if remap_tokens is None:
        remap_tokens = not rtl
    if remap_tokens:
        toks, scrs = remap_rtl_indices(toks, scrs)
    if hide_anchors:
        pairs = [(t, s) for t, s in zip(toks, scrs) if not is_anchor(t)]
        if pairs:
            toks, scrs = zip(*pairs)
            toks, scrs = list(toks), list(scrs)
    toks, scrs = _merge_tokens_for_display(toks, scrs)
    normed = _normalize_scores(scrs)

    spans: list[str] = []
    for text_raw, raw, norm in zip(toks, scrs, normed):
        text = html.escape(text_raw)
        style = (
            f"background:rgba(255,128,0,{norm:.3f});"
            "padding:0 2px;border-radius:3px;"
            "display:inline;"
            "font-family:Tahoma,'Segoe UI',system-ui,sans-serif;"
        )
        spans.append(
            f'<bdi dir="rtl" style="{style}" title="score={raw:.4f}">{text}</bdi>'
        )

    direction = "rtl" if rtl else "ltr"
    align = "right" if rtl else "left"
    sep = " " if rtl else " "
    body = sep.join(spans)
    if rtl:
        body = _RLM + body
    block_style = _RTL_BLOCK_STYLE if rtl else f"direction:ltr;text-align:left;line-height:2;"
    parts: list[str] = []
    if title:
        parts.append(
            f'<h4 dir="{direction}" style="margin:0 0 6px 0;text-align:{align};">'
            f"{html.escape(title)}</h4>"
        )
    parts.append(
        f'<div class="fa-rtl" dir="{direction}" lang="fa" style="{block_style}">{body}</div>'
    )
    return "\n".join(parts)


def write_html(path, tokens: Sequence[str], scores: Sequence[float], *, title: str | None = None) -> None:
    """Convenience: wrap the fragment in a minimal HTML doc and write it to disk."""
    from pathlib import Path

    fragment = render_text_heatmap(tokens, scores, title=title)
    doc = (
        "<!doctype html>\n"
        '<html lang="fa">\n<head>\n'
        '<meta charset="utf-8"><title>FA-GDCNet attention</title>\n'
        "</head>\n<body>\n"
        f"{fragment}\n"
        "</body>\n</html>\n"
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
