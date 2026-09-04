"""Tests for browser hashtag collector cookie normalization."""

from __future__ import annotations

import time
from types import SimpleNamespace

from data.collect_browser import (
    _playwright_expires,
    _to_playwright_cookie,
    caption_from_media_node,
    json_caption_worth_fetching,
    shortcodes_from_payload,
    shortcodes_from_response_text,
)


def test_playwright_expires_session_and_invalid() -> None:
    assert _playwright_expires(None) == -1
    assert _playwright_expires(0) == -1
    assert _playwright_expires(0.0) == -1
    assert _playwright_expires(-1) == -1
    assert _playwright_expires("0") == -1
    assert _playwright_expires("not-a-number") == -1


def test_playwright_expires_future() -> None:
    future = int(time.time()) + 3600
    assert _playwright_expires(future) == future
    assert _playwright_expires(float(future) + 0.9) == future
    # Firefox millisecond timestamps → seconds for Playwright.
    assert _playwright_expires(1813686522825) == 1813686522


def test_to_playwright_cookie_maps_session_expires() -> None:
    cookie = SimpleNamespace(
        name="sessionid",
        value="abc",
        domain=".instagram.com",
        path="/",
        expires=0,
        secure=True,
        same_site=0,
        _rest={},
    )
    pw = _to_playwright_cookie(cookie)
    assert pw["expires"] == -1
    assert pw["sameSite"] == "None"


def test_shortcodes_from_nested_graphql() -> None:
    payload = {
        "data": {
            "xdt_shortcode_media": None,
            "items": [
                {"code": "DOtAD_sDMFY", "product_type": "feed"},
                {"node": {"shortcode": "Db_b0NAsxrs", "product_type": "clips"}},
                {"code": "NO"},
            ],
        }
    }
    pairs = shortcodes_from_payload(payload)
    assert ("DOtAD_sDMFY", "p") in pairs
    assert ("Db_b0NAsxrs", "reel") in pairs
    assert all(code != "NO" for code, _ in pairs)


def test_shortcodes_from_response_text_regex_fallback() -> None:
    body = 'for (;;);{"code":"AbCdEfGhIjK","shortcode":"XyZ12345_ab"}'
    pairs = dict(shortcodes_from_response_text(body))
    assert "AbCdEfGhIjK" in pairs
    assert "XyZ12345_ab" in pairs


def test_json_caption_drops_english_before_fetch() -> None:
    assert json_caption_worth_fetching("just a selfie lol", sarcasm_candidates=True) is False
    assert json_caption_worth_fetching(None, sarcasm_candidates=True) is True
    persian_irony = "حالم بده ولی دارم میخندم"
    assert json_caption_worth_fetching(persian_irony, sarcasm_candidates=True) is True
    political = "طنز تلخ قطعی برق #خمینی #انقلاب"
    assert json_caption_worth_fetching(political, sarcasm_candidates=True) is False


def test_caption_from_media_node_shapes() -> None:
    assert caption_from_media_node({"caption": {"text": "hello"}}) == "hello"
    assert (
        caption_from_media_node(
            {"edge_media_to_caption": {"edges": [{"node": {"text": "inner"}}]}}
        )
        == "inner"
    )
