"""FA-GDCNet Streamlit UI (imported by scripts/fa_gdcnet_dashboard.py)."""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from data.schema import iter_dataset
from explain.render_image import overlay
from explain.render_text import render_text_heatmap
from explain.rollout import attention_from_image, attention_from_text
from inference.pipeline import Pipeline
from inference.smolvlm_check import is_smolvlm_pipeline, smolvlm_can_caption

DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
EXPLAIN_DIR = Path("reports") / "explain"
UI_BUILD = "2026-05-26-v13"

_RTL_PAGE_CSS = """
<style>
    [data-testid="stMarkdownContainer"] .fa-rtl,
    [data-testid="stMarkdownContainer"] div[dir="rtl"],
    .fa-rtl {
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: embed;
    }
    [data-testid="stMarkdownContainer"] .fa-rtl bdi {
        direction: rtl;
        unicode-bidi: isolate;
    }
    .fa-caption-label {
        margin: 0 0 0.25rem 0;
        font-weight: 600;
        direction: rtl;
        text-align: right;
    }
</style>
"""

_RTL_BLOCK_STYLE = (
    "direction:rtl;text-align:right;unicode-bidi:plaintext;"
    "font-family:Tahoma,'Segoe UI',system-ui,sans-serif;"
)


def _rtl_text_block(text: str, *, label: str | None = None) -> str:
    body = "\u200f" + html.escape(text)
    label_html = (
        f'<p class="fa-caption-label" dir="rtl" lang="fa">{html.escape(label)}</p>'
        if label
        else ""
    )
    return (
        f'<div class="fa-rtl" dir="rtl" lang="fa">'
        f"{label_html}"
        f'<p style="{_RTL_BLOCK_STYLE}margin:0;">{body}</p>'
        f"</div>"
    )


@st.cache_resource(show_spinner="Loading FA-GDCNet models (first run may take minutes)...")
def _cached_pipeline(*, _build: str = UI_BUILD) -> Pipeline:
    """`_build` bumps Streamlit cache when inference code changes."""
    pipeline = Pipeline.from_pretrained()
    smol = pipeline.bundle.smolvlm_model
    if is_smolvlm_pipeline(smol):
        label = f"{type(smol).__name__} → {type(smol.model).__name__}"
    else:
        label = type(smol).__name__
    st.sidebar.caption(f"SmolVLM: {label}")
    if not smolvlm_can_caption(smol):
        raise RuntimeError(
            "SmolVLM captioner cannot generate text. "
            "Run: .venv\\Scripts\\python.exe -m streamlit cache clear, then tasks.py dashboard"
        )
    return pipeline


@st.cache_data(show_spinner=False)
def _dataset_index() -> list[dict]:
    if not DATASET.exists():
        return []
    return [
        {"post_id": r.post_id, "caption": r.caption, "image_path": r.image_path, "label": r.label}
        for r in iter_dataset(DATASET)
    ]


def run_dashboard_app() -> None:
    st.set_page_config(page_title="FA-GDCNet Explainability", layout="wide")
    st.markdown(_RTL_PAGE_CSS, unsafe_allow_html=True)
    st.markdown(
        '<h1 dir="rtl" lang="fa" class="fa-rtl" style="text-align:right;">'
        "FA-GDCNet — تحلیل احساسات چندوجهی فارسی</h1>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"UI build {UI_BUILD}")

    index = _dataset_index()
    if not index:
        st.warning(f"Labeled dataset not found at {DATASET}. Run scrape + label first.")
        return

    post_ids = [row["post_id"] for row in index]
    selected = st.sidebar.selectbox("post_id", post_ids)
    row = next(r for r in index if r["post_id"] == selected)

    pipeline = _cached_pipeline()

    from PIL import Image

    image = Image.open(row["image_path"]).convert("RGB")
    features = pipeline.features_for(row["caption"], image)
    prediction = pipeline.predict_from_features(features)

    col_a, col_b = st.columns(2)
    with col_a:
        st.image(image, caption=row["image_path"], use_container_width=True)
    with col_b:
        st.markdown(_rtl_text_block(row["caption"], label="کپشن (فارسی)"), unsafe_allow_html=True)
        st.markdown(f"**Gold label:** `{row['label']}`")
        st.markdown(f"**Predicted:** `{prediction.label}` ({prediction.confidence:.3f})")
        if prediction.low_fidelity:
            st.markdown(
                f'<div class="fa-rtl" dir="rtl" lang="fa" style="{_RTL_BLOCK_STYLE}'
                "padding:0.75rem;background:#ffebee;border-radius:0.35rem;margin-top:0.5rem;\">"
                "⚠️ <strong>low_fidelity=True</strong> — توصیف تولیدشده توسط SmolVLM "
                "با تصویر همخوانی پایینی دارد؛ این پیش‌بینی را با احتیاط تفسیر کنید."
                "</div>",
                unsafe_allow_html=True,
            )
        st.json(prediction.discrepancy_vector)

    st.divider()
    st.markdown(
        '<h3 dir="rtl" lang="fa" class="fa-rtl" style="text-align:right;">'
        "توجه متنی (RTL)</h3>",
        unsafe_allow_html=True,
    )
    tokens, scores = attention_from_text(pipeline.bundle, row["caption"])
    st.markdown(render_text_heatmap(tokens, scores, title=None), unsafe_allow_html=True)

    st.subheader("Image attention overlay")
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    grid = attention_from_image(pipeline.bundle, image)
    overlay_path = overlay(image, grid, EXPLAIN_DIR / f"{selected}.png")
    st.image(str(overlay_path), use_container_width=True)
