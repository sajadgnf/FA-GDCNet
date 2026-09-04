"""Lightweight face-presence check for scraped Instagram images."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

_CASCADE_FILES = (
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_profileface.xml",
)


@lru_cache(maxsize=1)
def _face_cascades():
    import cv2

    loaded = []
    for name in _CASCADE_FILES:
        path = cv2.data.haarcascades + name
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            raise RuntimeError(f"failed to load OpenCV face cascade from {path}")
        loaded.append(cascade)
    return tuple(loaded)


def has_face(
    image_path: Path,
    *,
    min_size: int = 40,
    min_rel_area: float = 0.012,
) -> bool:
    """Return True if a face covers enough of the frame.

    Uses frontal and profile Haar cascades. ``min_rel_area`` still drops tiny
    false-positives on quote-cards / text graphics.
    """
    import cv2

    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"))
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    h, w = gray.shape[:2]
    area = float(w * h) or 1.0
    min_px = max(24, int(min_size))
    for cascade in _face_cascades():
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=3,
            minSize=(min_px, min_px),
        )
        if any((int(fw) * int(fh)) / area >= min_rel_area for _x, _y, fw, fh in faces):
            return True
    return False
