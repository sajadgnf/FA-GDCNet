"""Lightweight face-presence check for scraped Instagram images."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


@lru_cache(maxsize=1)
def _face_cascade():
    import cv2

    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"failed to load OpenCV face cascade from {path}")
    return cascade


def has_face(image_path: Path, *, min_size: int = 40) -> bool:
    """Return True if at least one frontal face is detected in the image."""
    import cv2

    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade().detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(min_size, min_size),
    )
    return len(faces) > 0
