"""Snapshot-style tests for the Persian HTML heatmap renderer."""

from __future__ import annotations

import re
from pathlib import Path

from explain.render_text import render_text_heatmap, write_html


def test_render_emits_rtl_direction():
    html = render_text_heatmap(["سلام", "دنیا"], [0.4, 0.7])
    assert 'dir="rtl"' in html


def test_render_one_bdi_per_word():
    tokens = ["سلام", "دنیا", "خوبی"]
    scores = [0.1, 0.5, 0.9]
    html = render_text_heatmap(tokens, scores)
    assert html.count("<bdi") == len(tokens)


def test_render_merges_trailing_punctuation():
    html = render_text_heatmap(["زیبایی", "!"], [0.1, 0.2])
    assert html.count("<bdi") == 1
    assert "زیبایی!" in html or "زیبایی!" in html.replace(" ", "")


def test_render_each_span_has_score_title():
    html = render_text_heatmap(["a", "b"], [0.1234, 0.5678])
    assert "score=0.1234" in html
    assert "score=0.5678" in html


def test_render_strips_wordpiece_prefix():
    html = render_text_heatmap(["##ها", "ها"], [0.1, 0.2])
    # `##ها` should render as just `ها` (no `##` in body).
    assert "##" not in html


def test_render_strips_sentencepiece_prefix():
    html = render_text_heatmap(["▁سلام"], [0.5])
    assert "▁" not in html


def test_render_with_ltr_disabled_remap():
    html_ltr = render_text_heatmap(["a", "b", "c"], [1, 2, 3], rtl=False, remap_tokens=False)
    assert 'dir="ltr"' in html_ltr
    # In ltr mode the natural left-to-right order is preserved.
    a_pos = html_ltr.find(">a</bdi>")
    c_pos = html_ltr.find(">c</bdi>")
    assert 0 <= a_pos < c_pos


def test_render_with_rtl_keeps_logical_token_order():
    html_rtl = render_text_heatmap(["a", "b", "c"], [1, 2, 3], rtl=True)
    a_pos = html_rtl.find(">a</bdi>")
    c_pos = html_rtl.find(">c</bdi>")
    assert 0 <= a_pos < c_pos


def test_render_with_ltr_remap_reverses_token_order():
    html = render_text_heatmap(["a", "b", "c"], [1, 2, 3], rtl=False, remap_tokens=True)
    assert html.find(">a</bdi>") > html.find(">c</bdi>")


def test_render_hides_anchor_tokens():
    html = render_text_heatmap(["[CLS]", "سلام", "[SEP]"], [0.0, 0.5, 0.0])
    assert "[CLS]" not in html
    assert "[SEP]" not in html
    assert "سلام" in html


def test_render_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        render_text_heatmap(["a", "b"], [0.1])


def test_render_title_is_html_escaped():
    html = render_text_heatmap(["a"], [0.5], title="<script>alert(1)</script>")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_write_html_creates_file(tmp_path: Path):
    out = tmp_path / "sub" / "x.html"
    write_html(out, ["a", "b"], [0.1, 0.2], title="title")
    contents = out.read_text(encoding="utf-8")
    assert "<!doctype html>" in contents.lower()
    assert "title" in contents


def test_normalised_opacity_within_unit_interval():
    html = render_text_heatmap(["a", "b", "c"], [-2.5, 0.5, 10.0])
    # Each rgba alpha must lie in [0, 1].
    alphas = [float(m.group(1)) for m in re.finditer(r"rgba\(255,128,0,([0-9.]+)\)", html)]
    for a in alphas:
        assert 0.0 <= a <= 1.0
