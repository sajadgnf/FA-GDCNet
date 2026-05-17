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

from .rtl import remap_rtl_indices

# WordPiece / SentencePiece prefixes used by typical Persian tokenizers.
_PIECE_PREFIXES = ("##",)
_SP_PREFIX = "▁"  # SentencePiece


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


def render_text_heatmap(
    tokens: Sequence[str],
    scores: Sequence[float],
    *,
    title: str | None = None,
    rtl: bool = True,
) -> str:
    """Return an HTML fragment with per-token attention highlighting.

    Per spec, scores are remapped (not modified) for RTL rendering when
    `rtl=True`. Set `rtl=False` to bypass the remap (useful for LTR captions).
    """
    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must have the same length")
    toks = list(tokens)
    scrs = [float(s) for s in scores]
    if rtl:
        toks, scrs = remap_rtl_indices(toks, scrs)
    normed = _normalize_scores(scrs)

    spans: list[str] = []
    for tok, raw, norm in zip(toks, scrs, normed):
        text = html.escape(_clean_token(tok))
        # `rgba` with a fixed warm hue; opacity = normalised score.
        style = (
            f"background:rgba(255,128,0,{norm:.3f});"
            "padding:1px 3px;border-radius:3px;margin:0 1px;"
            "font-family:Tahoma,'Iran Sans',sans-serif;"
        )
        spans.append(
            f'<span style="{style}" title="score={raw:.4f}">{text}</span>'
        )

    direction = "rtl" if rtl else "ltr"
    body = "".join(spans)
    parts: list[str] = []
    if title:
        parts.append(f'<h4 style="margin:0 0 6px 0;">{html.escape(title)}</h4>')
    parts.append(f'<div dir="{direction}" style="line-height:2;">{body}</div>')
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
