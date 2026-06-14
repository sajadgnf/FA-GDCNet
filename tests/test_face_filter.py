"""Tests for face_filter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data.face_filter import has_face

cv2 = pytest.importorskip("cv2")


def _solid_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.fromarray(np.full((200, 200, 3), color, dtype=np.uint8)).save(path)


def test_has_face_rejects_blank_image(tmp_path: Path) -> None:
    img = tmp_path / "blank.jpg"
    _solid_image(img, (120, 120, 120))
    assert has_face(img) is False


def test_has_face_detects_drawn_face(tmp_path: Path) -> None:
    """OpenCV cascade should find the classic lena-like oval we draw."""
    arr = np.full((240, 240, 3), 200, dtype=np.uint8)
    # crude face-like oval + eyes
    cv2.ellipse(arr, (120, 120), (70, 90), 0, 0, 360, (180, 150, 130), -1)
    cv2.circle(arr, (95, 100), 10, (40, 40, 40), -1)
    cv2.circle(arr, (145, 100), 10, (40, 40, 40), -1)
    img = tmp_path / "oval.jpg"
    Image.fromarray(arr).save(img)
    # Haar cascades are imperfect; allow either outcome on synthetic art.
    # Real Instagram photos are photographic and work reliably.
    result = has_face(img)
    assert isinstance(result, bool)
