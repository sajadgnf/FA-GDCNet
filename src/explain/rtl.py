"""RTL token remap for visualization.

Per spec scenario *RTL token remap before rendering*: the attention scores
themselves are NEVER modified. Only the order of (token, score) pairs is
flipped so the right-to-left reading order matches the natural Persian flow
when the renderer lays tokens out left-to-right.

Edge cases handled:
- `[CLS]` / `[SEP]` / `</s>` / `<s>` (and BERT-style `[PAD]`) at the head/tail
  of the token list are preserved at their original positions. They are
  markers, not content, and reversing them would corrupt the visualization.
- The function is a no-op if the underlying score values are not changed —
  this is asserted by `tests/test_rtl.py`.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# Tokens that should keep their original position regardless of language.
_ANCHOR_TOKENS: frozenset[str] = frozenset(
    {
        "[CLS]",
        "[SEP]",
        "[PAD]",
        "[UNK]",
        "<s>",
        "</s>",
        "<pad>",
        "<unk>",
        "<|endoftext|>",
    }
)


def is_anchor(token: str) -> bool:
    return token in _ANCHOR_TOKENS


def remap_rtl_indices(
    tokens: Sequence[str],
    scores: Sequence[float],
) -> tuple[list[str], list[float]]:
    """Return (reversed_tokens, reversed_scores) preserving anchor positions.

    The list of scores is permuted, not modified value-wise. Tokens at the
    start and end of the sequence that are anchors stay put; the contiguous
    interior block is reversed.
    """
    if len(tokens) != len(scores):
        raise ValueError(
            f"remap_rtl_indices: length mismatch tokens={len(tokens)} scores={len(scores)}"
        )
    n = len(tokens)
    if n == 0:
        return [], []

    # Find the interior slice [start, end) that excludes leading/trailing anchors.
    start = 0
    while start < n and is_anchor(tokens[start]):
        start += 1
    end = n
    while end > start and is_anchor(tokens[end - 1]):
        end -= 1

    out_tokens = list(tokens)
    out_scores = list(scores)
    if end > start:
        out_tokens[start:end] = list(reversed(out_tokens[start:end]))
        out_scores[start:end] = list(reversed(out_scores[start:end]))
    return out_tokens, out_scores


def remap_rtl_pairs(pairs: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    """Tuple-pair convenience wrapper around `remap_rtl_indices`."""
    tokens: list[str] = []
    scores: list[float] = []
    for t, s in pairs:
        tokens.append(t)
        scores.append(float(s))
    rt, rs = remap_rtl_indices(tokens, scores)
    return list(zip(rt, rs))
