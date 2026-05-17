"""Alpha-blended overlay of a patch-level attention map on the original image.

Output is a PNG saved to `reports/explain/<post_id>.png` per the spec.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def overlay(
    image,
    attention_grid: np.ndarray,
    out_path: str | Path,
    *,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Path:
    """Save a PNG of `image` with `attention_grid` alpha-blended on top.

    Args:
        image: PIL Image or numpy HWC array.
        attention_grid: 2-D `(G, G)` patch-level attention map.
        out_path: destination PNG path.
        alpha: blend factor for the heatmap layer.
        colormap: a matplotlib colormap name.

    Returns:
        The resolved `Path` written to.
    """
    from PIL import Image  # lazy
    from matplotlib import colormaps

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, np.ndarray):
        pil = Image.fromarray(image.astype(np.uint8))
    else:
        pil = image.convert("RGB")

    width, height = pil.size
    grid = _normalize(np.asarray(attention_grid, dtype=np.float32))
    heat_img = Image.fromarray((grid * 255).astype(np.uint8)).resize(
        (width, height), resample=Image.BILINEAR
    )
    heat = np.asarray(heat_img, dtype=np.float32) / 255.0

    cmap = colormaps.get_cmap(colormap)
    colored = cmap(heat)[..., :3]  # drop alpha channel
    base = np.asarray(pil, dtype=np.float32) / 255.0
    blended = (1.0 - alpha) * base + alpha * colored
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(blended).save(out)
    return out
