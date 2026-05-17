"""Streamlit explainability dashboard.

Spec scenarios (`Explainability Dashboard`):
- Sidebar sample selection by `post_id`.
- Caption + image + prediction + discrepancy vector + both heatmaps.
- Low-fidelity banner when `low_fidelity=true`.
- 2-second render budget on a typical laptop CPU (achieved by caching the
  Pipeline construction and reusing it across selections).

Run with: `python tasks.py dashboard` (or `streamlit run src/explain/dashboard.py`).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from data.schema import iter_dataset
from explain.render_image import overlay
from explain.render_text import render_text_heatmap
from explain.rollout import attention_from_image, attention_from_text
from inference.pipeline import Pipeline

DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
EXPLAIN_DIR = Path("reports") / "explain"


@st.cache_resource(show_spinner="Loading FA-GDCNet backbones...")
def _load_pipeline() -> Pipeline:
    return Pipeline.from_pretrained()


@st.cache_data(show_spinner=False)
def _dataset_index() -> list[dict]:
    if not DATASET.exists():
        return []
    return [
        {"post_id": r.post_id, "caption": r.caption, "image_path": r.image_path, "label": r.label}
        for r in iter_dataset(DATASET)
    ]


def main() -> None:
    st.set_page_config(page_title="FA-GDCNet Explainability", layout="wide")
    st.title("FA-GDCNet — تحلیل احساسات چندوجهی فارسی")

    index = _dataset_index()
    if not index:
        st.warning(f"Labeled dataset not found at {DATASET}. Run scrape + label first.")
        return

    post_ids = [row["post_id"] for row in index]
    selected = st.sidebar.selectbox("post_id", post_ids)
    row = next(r for r in index if r["post_id"] == selected)

    pipeline = _load_pipeline()

    from PIL import Image

    image = Image.open(row["image_path"]).convert("RGB")
    features = pipeline.features_for(row["caption"], image)
    prediction = pipeline.predict_from_features(features)

    col_a, col_b = st.columns(2)
    with col_a:
        st.image(image, caption=row["image_path"], use_column_width=True)
    with col_b:
        st.markdown(f"**Caption (FA):** <div dir=\"rtl\">{row['caption']}</div>", unsafe_allow_html=True)
        st.markdown(f"**Gold label:** `{row['label']}`")
        st.markdown(f"**Predicted:** `{prediction.label}` ({prediction.confidence:.3f})")
        if prediction.low_fidelity:
            st.error(
                "⚠️ low_fidelity=True — توصیف تولیدشده توسط SmolVLM با تصویر همخوانی پایینی دارد؛ "
                "این پیش‌بینی را با احتیاط تفسیر کنید."
            )
        st.json(prediction.discrepancy_vector)

    st.divider()
    st.subheader("Text attention (RTL-aware)")
    tokens, scores = attention_from_text(pipeline.bundle, row["caption"])
    st.markdown(render_text_heatmap(tokens, scores, title=None), unsafe_allow_html=True)

    st.subheader("Image attention overlay")
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    grid = attention_from_image(pipeline.bundle, image)
    overlay_path = overlay(image, grid, EXPLAIN_DIR / f"{selected}.png")
    st.image(str(overlay_path), use_column_width=True)


if __name__ == "__main__":
    main()
