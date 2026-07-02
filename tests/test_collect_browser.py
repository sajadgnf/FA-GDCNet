"""Tests for browser hashtag collector cookie normalization."""

from __future__ import annotations

import time
from types import SimpleNamespace

from data.collect_browser import _playwright_expires, _to_playwright_cookie


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
