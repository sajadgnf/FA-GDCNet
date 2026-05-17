"""Explainability: Attention Rollout, RTL remap, heatmap rendering, dashboard."""

from .render_text import render_text_heatmap
from .rtl import remap_rtl_indices

__all__ = ["remap_rtl_indices", "render_text_heatmap"]
