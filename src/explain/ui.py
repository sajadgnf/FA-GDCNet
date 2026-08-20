"""FA-GDCNet Streamlit dashboard — cool dark AI-agent workspace.

Layout inspired by Progra.ai–style AI agent product UIs
(https://dribbble.com/shots/27659643-Ai-agent-website-Progra-ai):
slim rail, hero greeting, soft glass cards, pill composer.
Palette stays cool (navy / cyan / indigo) — no warm coral/amber.
Typography: local Vazir v16.1.0.
"""

from __future__ import annotations

import base64
import html
from functools import lru_cache
from pathlib import Path

import streamlit as st

from data.schema import LABELS, iter_dataset
from explain.render_image import overlay
from explain.render_text import render_text_heatmap
from explain.rollout import attention_from_image, attention_from_text
from inference.gdrm import FEATURE_NAMES
from inference.pipeline import Pipeline
from inference.smolvlm_check import is_smolvlm_pipeline, smolvlm_can_caption

DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
EXPLAIN_DIR = Path("reports") / "explain"
FONTS_DIR = Path(__file__).resolve().parents[2] / "static" / "fonts"
UI_BUILD = "2026-08-20-result-polish2"

LABEL_FA: dict[str, str] = {
    "positive": "مثبت",
    "negative": "منفی",
    "neutral": "خنثی",
    "positive_sarcasm": "کنایه مثبت",
    "negative_sarcasm": "کنایه منفی",
}

# Cool-only semantic accents (no amber/coral fills)
LABEL_COLOR: dict[str, str] = {
    "positive": "#34d399",
    "negative": "#fb7185",
    "neutral": "#94a3b8",
    "positive_sarcasm": "#38bdf8",
    "negative_sarcasm": "#a78bfa",
}

FEATURE_FA: dict[str, str] = {
    "Dsem": "اختلاف معنایی",
    "Dsen": "اختلاف احساسی",
    "Fvt": "وفاداری تصویر–متن",
    "cos_TI": "شباهت متن–تصویر",
    "polarity_T": "قطبیت متن",
    "polarity_T_hat": "قطبیت توصیف",
}

FEATURE_HINT: dict[str, str] = {
    "Dsem": "فاصلهٔ معنایی متن فارسی و توصیف تصویر",
    "Dsen": "تضاد قطبیت احساسی متن و توصیف",
    "Fvt": "هم‌راستایی تصویر با توصیف SmolVLM",
    "cos_TI": "شباهت کسینوسی متن و تصویر",
    "polarity_T": "امتیاز احساس متن (−۱ تا +۱)",
    "polarity_T_hat": "امتیاز احساس توصیف تصویر",
}

AGENT_CARDS = (
    ("✦", "تحلیل احساس", "پنج‌کلاسه: مثبت، منفی، خنثی و دو نوع کنایه"),
    ("◎", "شاخص‌های GDRM", "Dsem · Dsen · Fvt روی اختلاف متن و تصویر"),
    ("▣", "توضیح‌پذیری", "نقشهٔ توجه متنی و بصری"),
)


@lru_cache(maxsize=1)
def _vazir_font_face_css() -> str:
    faces: list[tuple[int, str]] = (
        (300, "Vazir-Light"),
        (400, "Vazir"),
        (500, "Vazir-Medium"),
        (700, "Vazir-Bold"),
    )
    chunks: list[str] = []
    for weight, stem in faces:
        woff2 = FONTS_DIR / f"{stem}.woff2"
        woff = FONTS_DIR / f"{stem}.woff"
        src_parts: list[str] = []
        if woff2.is_file():
            b64 = base64.b64encode(woff2.read_bytes()).decode("ascii")
            src_parts.append(f"url(data:font/woff2;base64,{b64}) format('woff2')")
        if woff.is_file():
            b64 = base64.b64encode(woff.read_bytes()).decode("ascii")
            src_parts.append(f"url(data:font/woff;base64,{b64}) format('woff')")
        if not src_parts:
            continue
        chunks.append(
            "@font-face{"
            "font-family:'Vazir';"
            "font-style:normal;"
            f"font-weight:{weight};"
            "font-display:swap;"
            f"src:{','.join(src_parts)};"
            "}"
        )
    if (FONTS_DIR / "Vazir-Medium.woff2").is_file():
        b64 = base64.b64encode((FONTS_DIR / "Vazir-Medium.woff2").read_bytes()).decode("ascii")
        chunks.append(
            "@font-face{font-family:'Vazir';font-style:normal;font-weight:600;"
            "font-display:swap;"
            f"src:url(data:font/woff2;base64,{b64}) format('woff2');}}"
        )
    return "\n".join(chunks)


_PAGE_CSS = """
<style>
/*VAZIR_FONTS*/

:root {
  color-scheme: dark;
  --bg: #070b14;
  --bg-elev: #0d1526;
  --surface: rgba(18, 28, 48, 0.72);
  --surface-solid: #121c30;
  --stroke: rgba(148, 163, 184, 0.14);
  --stroke-strong: rgba(56, 189, 248, 0.28);
  --text: #f1f5f9;
  --muted: #94a3b8;
  --cyan: #38bdf8;
  --indigo: #818cf8;
  --mint: #34d399;
  --grad: linear-gradient(120deg, #38bdf8 0%, #6366f1 55%, #a78bfa 100%);
  --glow: 0 0 80px rgba(56, 189, 248, 0.12), 0 0 120px rgba(99, 102, 241, 0.08);
  --font: 'Vazir', Tahoma, 'Segoe UI', sans-serif;
}

html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
  background:
    radial-gradient(900px 420px at 50% -8%, rgba(56, 189, 248, 0.16), transparent 60%),
    radial-gradient(700px 380px at 85% 20%, rgba(99, 102, 241, 0.12), transparent 55%),
    radial-gradient(600px 360px at 10% 40%, rgba(167, 139, 250, 0.08), transparent 50%),
    var(--bg) !important;
  color: var(--text) !important;
  color-scheme: dark;
  font-family: var(--font) !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }

[data-testid="stSidebar"] {
  background: #050814 !important;
  border-inline-end: 1px solid var(--stroke) !important;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] label,
button, input, textarea, p, h1, h2, h3, span, div {
  font-family: var(--font) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stCaption { color: var(--muted) !important; }
[data-testid="stSidebar"] label { direction: rtl; text-align: right; }
[data-testid="stSidebar"] [data-baseweb="radio"] { direction: rtl; width: 100%; }

.block-container {
  max-width: 1000px !important;
  padding: 1.5rem 1.5rem 5rem !important;
}
[data-testid="stVerticalBlock"] { gap: 0.65rem !important; }

/* Hero */
.g-hello {
  text-align: center;
  padding: 2rem 0.5rem 1.25rem;
  position: relative;
}
.g-hello::before {
  content: "";
  position: absolute;
  inset: 10% 20% auto;
  height: 8rem;
  background: radial-gradient(ellipse, rgba(56,189,248,.18), transparent 70%);
  filter: blur(12px);
  pointer-events: none;
  z-index: 0;
}
.g-hello > * { position: relative; z-index: 1; }
.g-spark {
  font-size: 3.35rem;
  line-height: 1;
  margin-bottom: 0.85rem;
  background: var(--grad);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  display: inline-block;
}
.g-hello .g-title {
  margin: 0;
  font-size: 1.85rem;
  font-weight: 600;
  direction: rtl;
  color: var(--text);
  letter-spacing: -0.02em;
  line-height: 1.4;
}
.g-hello .grad {
  background: var(--grad);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.g-hello p {
  margin: 0.7rem auto 0;
  color: var(--muted);
  direction: rtl;
  font-size: 0.92rem;
  line-height: 1.7;
}

/* Agent / capability cards (Progra-style grid) */
.g-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.75rem;
  margin: 0.5rem 0 1.35rem;
  direction: rtl;
}
@media (max-width: 720px) {
  .g-grid { grid-template-columns: 1fr; }
}
.g-card {
  background: var(--surface);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid var(--stroke);
  border-radius: 1.1rem;
  padding: 1rem 1.05rem;
  text-align: right;
  transition: border-color .2s ease, box-shadow .2s ease;
}
.g-card:hover {
  border-color: var(--stroke-strong);
  box-shadow: 0 0 0 1px rgba(56,189,248,.12), var(--glow);
}
.g-card .ico {
  width: 2.5rem; height: 2.5rem;
  border-radius: 0.75rem;
  display: grid; place-items: center;
  background: rgba(56, 189, 248, 0.12);
  color: var(--cyan);
  font-size: 1.25rem;
  margin-bottom: 0.7rem;
  margin-inline-start: auto;
}
.g-card .g-card-title {
  margin: 0 0 0.35rem;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text);
}
.g-card p {
  margin: 0;
  font-size: 0.95rem;
  line-height: 1.65;
  color: var(--muted);
}

/* Streamlit auto-link icons next to markdown headings */
.g-card a,
[data-testid="stMarkdownContainer"] a[href^="#"],
.stMarkdown a[href^="#"],
h1 > span > a, h2 > span > a, h3 > span > a,
[data-testid="stHeadingWithActionElements"] a {
  display: none !important;
}

.g-chips {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.5rem;
  margin: 0 0 1.25rem;
  direction: rtl;
}
.g-chip {
  display: inline-block;
  padding: 0.6rem 1.1rem;
  border-radius: 999px;
  background: rgba(18, 28, 48, 0.85);
  border: 1px solid var(--stroke);
  color: var(--muted);
  font-size: 0.95rem;
  font-weight: 500;
  line-height: 1.35;
}

.g-hint {
  text-align: center;
  direction: rtl;
  color: var(--muted);
  font-size: 0.85rem;
  margin: 0 0 1rem;
  line-height: 1.55;
}

/* Conversation */
.g-user {
  background: var(--surface-solid);
  border: 1px solid var(--stroke);
  border-radius: 1.15rem;
  padding: 0.9rem 1.05rem;
  margin: 0.75rem 0 1.35rem;
  direction: rtl;
  text-align: right;
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
}
.g-user .body { margin: 0; font-size: 0.95rem; line-height: 1.7; color: var(--text); }

.g-bot-panel {
  background: var(--surface);
  backdrop-filter: blur(16px);
  border: 1px solid var(--stroke-strong);
  border-radius: 1.25rem;
  padding: 1.1rem 1.2rem;
  margin: 0.75rem 0 1.25rem;
  box-shadow: var(--glow);
  direction: rtl;
  text-align: right;
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
}
.g-bot-head {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  flex-direction: row-reverse;
  margin-bottom: 0.7rem;
}
.g-bot-head .spark {
  font-size: 1rem;
  background: var(--grad);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.g-bot-head .name { font-size: 0.8rem; font-weight: 500; color: var(--muted); }
.g-answer {
  font-size: 1.05rem;
  line-height: 1.75;
  color: var(--text);
  margin: 0 0 0.65rem;
}
.g-answer strong { font-weight: 600; color: var(--cyan); }
.g-pill {
  display: inline-block;
  padding: 0.28rem 0.75rem;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 600;
  color: #061018;
  margin-inline: 0.15rem;
}
.g-meta-line {
  color: var(--muted);
  font-size: 0.84rem;
  margin: 0.3rem 0;
  line-height: 1.55;
}
.g-meta-line code {
  background: rgba(56,189,248,.1);
  color: var(--cyan);
  padding: 0.1rem 0.35rem;
  border-radius: 0.3rem;
  font-size: 0.75rem;
}

.g-metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.55rem;
  direction: rtl;
  margin: 0.85rem 0 0.25rem;
}
@media (max-width: 640px) {
  .g-metrics { grid-template-columns: 1fr; }
}
.g-metric {
  background: rgba(7, 11, 20, 0.55);
  border: 1px solid var(--stroke);
  border-radius: 0.9rem;
  padding: 0.7rem 0.8rem;
  text-align: right;
}
.g-metric .k { color: var(--muted); font-size: 0.72rem; display: block; margin-bottom: 0.25rem; }
.g-metric .v {
  color: var(--cyan);
  font-size: 1.05rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  direction: ltr;
  text-align: left;
}

.g-note {
  background: rgba(7, 11, 20, 0.45);
  border: 1px solid var(--stroke);
  border-radius: 0.95rem;
  padding: 0.75rem 0.9rem;
  margin: 0.65rem 0 1.5rem;
  direction: rtl;
  text-align: right;
}
.g-note .lbl { color: var(--indigo); font-weight: 500; font-size: 0.75rem; margin-bottom: 0.3rem; }
.g-note .en {
  direction: ltr; text-align: left; color: var(--muted);
  font-size: 0.84rem; line-height: 1.5; margin: 0;
}
.g-warn {
  direction: rtl; text-align: right;
  padding: 0.65rem 0.85rem; border-radius: 0.85rem;
  background: rgba(251, 113, 133, 0.1);
  border: 1px solid rgba(251, 113, 133, 0.25);
  color: #fda4af; font-size: 0.85rem; line-height: 1.55; margin: 0.5rem 0;
}
.g-ok {
  direction: rtl; text-align: right;
  padding: 0.55rem 0.85rem; border-radius: 0.85rem;
  background: rgba(52, 211, 153, 0.1);
  border: 1px solid rgba(52, 211, 153, 0.22);
  color: var(--mint); font-size: 0.84rem; margin: 0.5rem 0;
}
.g-section {
  direction: rtl; text-align: right;
  color: var(--muted); font-size: 0.78rem; font-weight: 500;
  margin: 1rem 0 0.45rem;
}
.g-heatmap {
  background: var(--surface-solid);
  border: 1px solid var(--stroke);
  border-radius: 1rem;
  padding: 0.85rem 1rem;
  direction: rtl;
  font-size: 0.92rem;
}

[data-testid="stTextArea"] {
  direction: rtl !important;
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  overflow: visible;
}
[data-testid="stTextArea"] > label {
  display: block !important;
  margin: 0 0 0.45rem !important;
  padding: 0 !important;
  color: var(--muted) !important;
  direction: rtl !important;
  text-align: right !important;
  font-size: 0.85rem !important;
  width: 100%;
  background: transparent !important;
  border: none !important;
}
/* Dashed shell only on the field wrapper under the label */
[data-testid="stTextArea"] > div {
  background: rgba(7, 11, 20, 0.55) !important;
  border: 1px dashed var(--stroke) !important;
  border-radius: 1rem !important;
  box-shadow: none !important;
  outline: none !important;
  overflow: hidden;
}
[data-testid="stTextArea"] > div div {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
}
[data-testid="stTextArea"] textarea {
  background: transparent !important;
  color: var(--text) !important;
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  font-size: 0.95rem !important;
  line-height: 1.65 !important;
  padding: 16px !important;
  min-height: 96px !important;
  direction: rtl !important;
  text-align: right !important;
  border-radius: 1rem !important;
}
[data-testid="stTextArea"] textarea:focus {
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
}
/* Hide "Press Ctrl+Enter to apply" (and similar) hints */
[data-testid="InputInstructions"],
[data-testid="stTextArea"] [data-testid="InputInstructions"],
.stTextArea [data-testid="InputInstructions"] {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  width: 0 !important;
  overflow: hidden !important;
  opacity: 0 !important;
  pointer-events: none !important;
}
[data-testid="stSelectbox"] label {
  color: var(--muted) !important;
  direction: rtl !important;
  text-align: right !important;
  font-size: 0.85rem !important;
  width: 100%;
  display: block !important;
  margin-bottom: 0.45rem !important;
}
[data-testid="stFileUploader"] > label {
  color: var(--muted) !important;
  direction: rtl !important;
  text-align: right !important;
  font-size: 0.85rem !important;
  width: 100%;
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  justify-content: flex-start !important;
  gap: 0.4rem !important;
  margin-bottom: 0.45rem !important;
  flex-wrap: nowrap !important;
}
[data-testid="stFileUploader"] > label > span,
[data-testid="stFileUploader"] > label > div {
  display: inline-flex !important;
  align-items: center !important;
  margin: 0 !important;
  width: auto !important;
  max-width: none !important;
}
[data-testid="stFileUploader"] {
  direction: rtl !important;
}
[data-testid="stFileUploaderDropzone"] {
  background: rgba(7, 11, 20, 0.55) !important;
  border: 1px dashed var(--stroke) !important;
  border-radius: 1rem !important;
  padding: 16px !important;
  min-height: 72px !important;
  direction: rtl !important;
  text-align: right !important;
  cursor: pointer !important;
}
[data-testid="stFileUploaderDropzone"] * {
  color: var(--muted) !important;
  font-size: 0.85rem !important;
}
/* Hide browse/upload button in dropzone (drag-drop + dropzone click still work) */
[data-testid="stFileUploaderDropzone"] button[kind="secondary"],
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"],
[data-testid="stFileUploaderDropzone"] > span:has(button) {
  display: none !important;
}
/* Hide English "Upload" label; keep the icon */
[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"],
[data-testid="stFileUploader"] button[kind="secondary"] p,
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] p {
  display: none !important;
}
/* Keep browse + delete buttons; only restyle */
[data-testid="stFileUploader"] button[kind="secondary"],
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
  border-radius: 0.75rem !important;
  border: 1px solid var(--stroke) !important;
  background: rgba(18, 28, 48, 0.9) !important;
  color: var(--text) !important;
  direction: rtl !important;
}
[data-testid="stFileUploaderDeleteBtn"],
[data-testid="stFileUploader"] button[title*="Remove"],
[data-testid="stFileUploader"] button[aria-label*="Remove"],
[data-testid="stFileUploader"] button[aria-label*="Delete"] {
  display: inline-flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  color: var(--muted) !important;
}
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFile"] {
  direction: rtl !important;
  color: var(--text) !important;
}
/* Single-file mode: hide Streamlit's "Add files" control after upload */
[data-testid="stFileChips"] > button,
[data-testid="stFileUploader"] button[aria-label="Add files"],
[data-testid="stFileUploader"] button[aria-label*="Add file"],
[data-testid="stFileUploader"] button[title="Add files"],
[data-testid="stFileUploader"] button[aria-label="Add"],
[data-testid="stFileChips"] button[kind="borderlessIcon"],
[data-testid="stFileChips"] [data-testid="stBaseButton-borderlessIcon"] {
  display: none !important;
  pointer-events: none !important;
  width: 0 !important;
  height: 0 !important;
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
}
.g-toast {
  position: fixed;
  bottom: 1.5rem;
  left: 50%;
  transform: translateX(-50%);
  z-index: 99999;
  background: rgba(18, 28, 48, 0.96);
  border: 1px solid rgba(251, 113, 133, 0.45);
  color: #fda4af;
  padding: 0.75rem 1.15rem;
  border-radius: 0.85rem;
  font-family: Vazir, sans-serif;
  font-size: 0.9rem;
  direction: rtl;
  box-shadow: 0 12px 32px rgba(0,0,0,.45);
  pointer-events: none;
}
[data-testid="stCaptionContainer"],
.stCaption {
  color: var(--muted) !important;
  direction: rtl !important;
  text-align: right !important;
  display: block;
  width: 100%;
}

.stButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] button[kind="primary"],
[data-testid="stFormSubmitButton"] > button,
[data-testid="stFormSubmitButton"] button {
  background: var(--grad) !important;
  background-image: var(--grad) !important;
  background-color: #38bdf8 !important;
  color: #061018 !important;
  border: none !important;
  border-radius: 999px !important;
  padding: 0.55rem 1.5rem !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  box-shadow: 0 8px 24px rgba(56, 189, 248, 0.25) !important;
}
.stButton > button[kind="primary"]:hover:not(:disabled),
[data-testid="stFormSubmitButton"] button:hover:not(:disabled) {
  filter: brightness(1.08);
}
.stButton > button[kind="primary"]:disabled,
.stButton > button:disabled,
[data-testid="stFormSubmitButton"] button:disabled {
  opacity: 0.85 !important;
  cursor: wait !important;
  pointer-events: none !important;
  filter: none !important;
  box-shadow: 0 8px 24px rgba(56, 189, 248, 0.2) !important;
}
[data-testid="stButton"] { display: flex; justify-content: center; margin-top: 0.45rem; }
[data-testid="stFormSubmitButton"] { display: flex; justify-content: center; margin-top: 0.45rem; }
[data-testid="stForm"] { border: none !important; padding: 0 !important; }

/* In-button loading: spinner + label */
[data-testid="stFormSubmitButton"] button.g-busy-btn,
[data-testid="stFormSubmitButton"] button:disabled {
  gap: 0.5rem !important;
}
[data-testid="stFormSubmitButton"] button:disabled::before {
  content: "";
  width: 0.95rem;
  height: 0.95rem;
  border-radius: 50%;
  border: 2px solid rgba(6, 16, 24, 0.25);
  border-top-color: #061018;
  animation: g-spin 0.7s linear infinite;
  flex-shrink: 0;
}
@keyframes g-spin {
  to { transform: rotate(360deg); }
}

[data-testid="stExpander"] {
  background: var(--surface-solid) !important;
  border: 1px solid var(--stroke) !important;
  border-radius: 1rem !important;
  direction: rtl !important;
  text-align: right !important;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary {
  direction: rtl !important;
  text-align: right !important;
}
[data-testid="stExpander"] summary {
  display: flex !important;
  flex-direction: row-reverse !important;
  align-items: center !important;
  justify-content: flex-end !important;
  gap: 0.5rem !important;
}
[data-testid="stExpander"] summary > div,
[data-testid="stExpander"] summary p {
  direction: rtl !important;
  text-align: right !important;
  width: 100%;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"],
[data-testid="stExpander"] table,
[data-testid="stExpander"] thead,
[data-testid="stExpander"] tbody,
[data-testid="stExpander"] tr,
[data-testid="stExpander"] th,
[data-testid="stExpander"] td {
  direction: rtl !important;
  text-align: right !important;
}
[data-testid="stExpander"] table {
  width: 100%;
}
/* Fix broken Material icon showing as "arrow_right" text */
[data-testid="stExpander"] [data-testid="stExpanderIcon"],
[data-testid="stExpander"] span[data-testid="stIconMaterial"],
[data-testid="stExpander"] .material-symbols-rounded,
[data-testid="stExpander"] [class*="material-symbols"] {
  font-family: inherit !important;
  font-size: 0 !important;
  width: 1.1rem !important;
  height: 1.1rem !important;
  display: inline-block !important;
  position: relative !important;
  overflow: hidden !important;
  color: transparent !important;
}
[data-testid="stExpander"] [data-testid="stExpanderIcon"]::before,
[data-testid="stExpander"] span[data-testid="stIconMaterial"]::before,
[data-testid="stExpander"] .material-symbols-rounded::before,
[data-testid="stExpander"] [class*="material-symbols"]::before {
  content: "▾" !important;
  font-size: 1rem !important;
  line-height: 1.1rem !important;
  color: var(--muted) !important;
  position: absolute !important;
  inset: 0 !important;
  display: grid !important;
  place-items: center !important;
}
[data-testid="stExpander"] details[open] [data-testid="stExpanderIcon"]::before,
[data-testid="stExpander"] details[open] span[data-testid="stIconMaterial"]::before,
[data-testid="stExpander"] details[open] .material-symbols-rounded::before,
[data-testid="stExpander"] details[open] [class*="material-symbols"]::before {
  content: "▴" !important;
}

[data-testid="stImage"] {
  width: 100% !important;
}
[data-testid="stImage"] img {
  border-radius: 1rem;
  max-height: 280px;
  object-fit: contain;
  width: 100%;
  background: var(--bg-elev);
  border: 1px solid var(--stroke);
  box-sizing: border-box;
}
[data-testid="stAlert"] {
  background: var(--surface-solid) !important;
  color: var(--text) !important;
  border: 1px solid var(--stroke) !important;
  border-radius: 1rem !important;
}
hr { border-color: var(--stroke) !important; opacity: 0.8 !important; }
.stCaption { color: var(--muted) !important; direction: rtl; text-align: right; }

.sb-title {
  direction: rtl; text-align: right;
  font-weight: 600; font-size: 0.9rem; color: var(--text);
  margin: 0.25rem 0 0.4rem;
}
.sb-body {
  direction: rtl; text-align: right;
  font-size: 0.8rem; line-height: 1.6; color: var(--muted);
}
.sb-body strong { color: var(--text); font-weight: 500; }
</style>
"""


def _page_css() -> str:
    return _PAGE_CSS.replace("/*VAZIR_FONTS*/", _vazir_font_face_css())


def _prediction_prose(
    label: str,
    confidence: float,
    *,
    polarity_T: float | None = None,
    polarity_T_hat: float | None = None,
) -> str:
    fa = LABEL_FA.get(label, label)
    pct = confidence * 100

    def _text_polarity_phrase(p: float | None) -> str:
        if p is None:
            return "متن از نظر قطبیت مبهم است"
        if p > 0.15:
            return "متن ظاهراً <strong>مثبت</strong> است"
        if p < -0.15:
            return "متن ظاهراً <strong>منفی</strong> است"
        return "متن از نظر قطبیت تقریباً <strong>خنثی</strong> است"

    templates = {
        "positive": "احساس غالب این پست را <strong>مثبت</strong> تشخیص دادم.",
        "negative": "احساس غالب این پست را <strong>منفی</strong> تشخیص دادم.",
        "neutral": "این پست را از نظر احساسی <strong>خنثی</strong> ارزیابی کردم.",
    }
    if label == "positive_sarcasm":
        body = (
            "نشانه‌های <strong>کنایهٔ مثبت‌نما</strong> دیده می‌شود — "
            f"{_text_polarity_phrase(polarity_T)}، اما با تصویر یا زمینه هم‌خوان نیست."
        )
    elif label == "negative_sarcasm":
        body = (
            "نشانه‌های <strong>کنایهٔ منفی‌نما</strong> دیده می‌شود — "
            f"{_text_polarity_phrase(polarity_T)}، اما با تصویر یا زمینه هم‌خوان نیست."
        )
    else:
        body = templates.get(label, f"برچسب پیش‌بینی‌شده: <strong>{html.escape(fa)}</strong>.")

    color = LABEL_COLOR.get(label, "#94a3b8")
    pill = f'<span class="g-pill" style="background:{color};">{html.escape(fa)}</span>'
    return (
        f'<p class="g-answer">{body}</p>'
        f'<p class="g-meta-line">برچسب: {pill} · اطمینان '
        f"<strong>{pct:.1f}</strong> · <code>{html.escape(label)}</code></p>"
    )


def _metric_cards(features: dict[str, float]) -> str:
    parts: list[str] = []
    for key in ("Dsem", "Dsen", "Fvt"):
        title = FEATURE_FA.get(key, key)
        val = float(features.get(key, 0.0))
        hint = html.escape(FEATURE_HINT.get(key, ""))
        parts.append(
            f'<div class="g-metric" title="{hint}">'
            f'<span class="k">{html.escape(title)} · {key}</span>'
            f'<div class="v">{val:.3f}</div></div>'
        )
    return f'<div class="g-metrics">{"".join(parts)}</div>'


def _hello() -> None:
    st.markdown(
        """
<div class="g-hello">
  <div class="g-spark">✦</div>
  <div class="g-title">تحلیل احساس و کنایه</div>
  <p>
    یک تصویر و متن فارسی بفرستید تا عامل <span class="grad">FA-GDCNet</span>
    احساس، کنایه و شاخص‌های اختلاف مولد را تحلیل کند.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )


def _capability_cards() -> None:
    html_cards = []
    for ico, title, desc in AGENT_CARDS:
        html_cards.append(
            f'<div class="g-card"><div class="ico">{ico}</div>'
            f'<div class="g-card-title">{html.escape(title)}</div>'
            f"<p>{html.escape(desc)}</p></div>"
        )
    st.markdown(f'<div class="g-grid">{"".join(html_cards)}</div>', unsafe_allow_html=True)


def _suggestion_chips(index: list[dict]) -> None:
    if not index:
        return
    by_label: dict[str, dict] = {}
    for row in index:
        by_label.setdefault(row["label"], row)
    chips = []
    for lab in LABELS:
        if lab not in by_label:
            continue
        chips.append(f'<span class="g-chip">نمونهٔ {html.escape(LABEL_FA[lab])}</span>')
    if chips:
        st.markdown(f'<div class="g-chips">{"".join(chips[:5])}</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="در حال آماده‌سازی")
def _cached_pipeline(*, _build: str = UI_BUILD) -> Pipeline:
    pipeline = Pipeline.from_pretrained()
    smol = pipeline.bundle.smolvlm_model
    if not smolvlm_can_caption(smol):
        raise RuntimeError(
            "SmolVLM captioner cannot generate text. "
            "Run: python -m streamlit cache clear, then python tasks.py dashboard"
        )
    if is_smolvlm_pipeline(smol):
        st.session_state["_smolvlm_label"] = f"{type(smol).__name__} → {type(smol.model).__name__}"
    else:
        st.session_state["_smolvlm_label"] = type(smol).__name__
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
    st.set_page_config(
        page_title="FA-GDCNet",
        page_icon="✦",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.markdown(_page_css(), unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<p class="sb-title">✦ فضای کار</p>', unsafe_allow_html=True)
        mode = st.radio(
            "حالت",
            ["گفتگو با عامل", "مرور نمونه‌های برچسب‌خورده"],
            index=0,
        )
        st.caption(f"نسخه {UI_BUILD}")
        if label := st.session_state.get("_smolvlm_label"):
            st.caption(label)
        st.divider()
        st.markdown(
            """
<div class="sb-body">
<strong>پنج کلاس خروجی</strong><br>
مثبت · منفی · خنثی<br>
کنایه مثبت · کنایه منفی<br><br>
مدل‌ها فریز هستند؛ فقط طبقه‌بند سبک sklearn آموزش دیده است.
</div>
""",
            unsafe_allow_html=True,
        )

    pipeline = _cached_pipeline()
    index = _dataset_index()

    if mode == "گفتگو با عامل":
        _hello()
        _capability_cards()
        _suggestion_chips(index)
        _render_composer(pipeline)
        return

    if not index:
        st.warning("دیتاست برچسب‌خورده یافت نشد. ابتدا scrape و label را اجرا کنید.")
        return

    with st.sidebar:
        st.divider()
        filter_label = st.selectbox(
            "فیلتر برچسب",
            ["همه"] + [LABEL_FA[l] for l in LABELS],
        )
        filtered = index
        if filter_label != "همه":
            key = next(k for k, v in LABEL_FA.items() if v == filter_label)
            filtered = [r for r in index if r["label"] == key]
        selected = st.selectbox("شناسهٔ پست", [r["post_id"] for r in filtered])

    row = next(r for r in filtered if r["post_id"] == selected)
    from PIL import Image

    image = Image.open(row["image_path"]).convert("RGB")
    with st.spinner("عامل در حال تحلیل است…"):
        _render_conversation(
            pipeline,
            caption=row["caption"],
            image=image,
            gold_label=row["label"],
            post_id=selected,
            show_attention=True,
        )


def _patch_uploader_fa() -> None:
    """Persianize uploader chrome + live-enable submit (image + ≥3 words)."""
    import streamlit.components.v1 as components

    components.html(
        """
<script>
(function () {
  const w = window.parent;
  const doc = w.document;
  if (w.__gFaUploaderInterval) {
    clearInterval(w.__gFaUploaderInterval);
    w.__gFaUploaderInterval = null;
  }

  function toast(msg) {
    let el = doc.getElementById('g-fa-toast');
    if (!el) {
      el = doc.createElement('div');
      el.id = 'g-fa-toast';
      el.className = 'g-toast';
      doc.body.appendChild(el);
    }
    el.textContent = msg;
    el.style.display = 'block';
    clearTimeout(el._t);
    el._t = setTimeout(() => { el.style.display = 'none'; }, 3200);
  }

  function rootEl() {
    return doc.querySelector('[data-testid="stFileUploader"]');
  }

  function hideAdd(root) {
    if (!root) return;
    root.querySelectorAll('button').forEach((btn) => {
      const label = (
        (btn.getAttribute('aria-label') || '') + ' ' +
        (btn.getAttribute('title') || '') + ' ' +
        (btn.textContent || '')
      ).toLowerCase();
      if (label.includes('add file') || label.trim() === 'add') {
        btn.style.setProperty('display', 'none', 'important');
      }
    });
    const chips = root.querySelector('[data-testid="stFileChips"]');
    if (!chips) return;
    Array.from(chips.children).forEach((child) => {
      if (child.tagName === 'BUTTON') {
        child.style.setProperty('display', 'none', 'important');
      }
    });
  }

  function localize(root) {
    if (!root) return;
    root.querySelectorAll('button').forEach((btn) => {
      const aria = btn.getAttribute('aria-label') || '';
      if (/delete|remove/i.test(aria)) {
        btn.setAttribute('aria-label', 'حذف');
        btn.setAttribute('title', 'حذف تصویر');
        return;
      }
      btn.querySelectorAll('p, span').forEach((el) => {
        const t = (el.textContent || '').trim().toLowerCase();
        if (t === 'upload' || t === 'browse files') el.textContent = '';
      });
    });
    root.querySelectorAll('span, small').forEach((el) => {
      if (el.closest('button')) return;
      const t = (el.textContent || '').trim();
      if (/MB per file/i.test(t) || (/per file/i.test(t) && /MB/i.test(t))) {
        el.textContent = 'حداکثر ۲۰۰ مگابایت برای هر فایل · JPG، PNG، WEBP';
      } else if (/Drag and drop/i.test(t)) {
        el.textContent = 'فایل را بکشید و اینجا رها کنید';
      }
    });
  }

  function pruneErrorChips(root) {
    if (!root) return;
    root.querySelectorAll('[data-testid="stFileChip"], [data-testid="stFileUploaderFile"]').forEach((chip) => {
      const text = chip.textContent || '';
      if (!/not allowed|Error:/i.test(text)) return;
      const btn = chip.querySelector('button');
      if (btn) {
        btn.click();
        toast('فقط فایل‌های JPG، PNG و WEBP پشتیبانی می‌شوند.');
      }
    });
  }

  function unlockSubmit() {
    // Never force-enable while the button is in loading label state.
    doc.querySelectorAll('button').forEach((btn) => {
      const t = (btn.textContent || '');
      if (!t.includes('ارسال') && !t.includes('بارگذاری')) return;
      if (t.includes('بارگذاری')) return;
    });
  }

  function patch() {
    const root = rootEl();
    if (root) {
      const input = root.querySelector('input[type="file"]');
      if (input) {
        input.multiple = false;
        input.removeAttribute('multiple');
      }
      hideAdd(root);
      localize(root);
      pruneErrorChips(root);
    }
    unlockSubmit();
  }

  w.__gFaUploaderPatch = patch;
  patch();
  w.__gFaUploaderInterval = setInterval(patch, 600);
})();
</script>
""",
        height=0,
    )


def _text_len(text: str) -> int:
    return len((text or "").strip())


def _render_composer(pipeline: Pipeline) -> None:
    from PIL import Image

    busy = bool(st.session_state.get("composer_busy", False))

    # Uploader outside any form so the file is available on submit.
    uploaded = st.file_uploader(
        "بارگذاری تصویر",
        type=["jpg", "jpeg", "png", "webp"],
        help="فایل را بکشید و رها کنید، یا ناحیه را کلیک کنید",
        accept_multiple_files=False,
        key="composer_image",
        disabled=busy,
    )
    _patch_uploader_fa()

    allowed_ext = {".jpg", ".jpeg", ".png", ".webp"}
    if uploaded is not None:
        suffix = Path(uploaded.name or "").suffix.lower()
        mime = (getattr(uploaded, "type", None) or "").lower()
        mime_ok = mime in {"image/jpeg", "image/jpg", "image/png", "image/webp"}
        if suffix not in allowed_ext and not mime_ok:
            st.error("فقط فایل‌های JPG، PNG و WEBP پشتیبانی می‌شوند.")
            uploaded = None

    with st.form("agent_composer", clear_on_submit=False, enter_to_submit=False, border=False):
        caption = st.text_area(
            "متن",
            height=112,
            placeholder="متن مورد نظر خود را بنویسید",
            key="composer_caption",
            disabled=busy,
        )
        st.caption("برای ارسال: یک تصویر + حداقل ۳ حرف")
        submit_label = "در حال بارگذاری" if busy else "ارسال"
        clicked = st.form_submit_button(submit_label, type="primary", disabled=busy)

    # Phase 2: run model while button shows loading, then unlock via rerun.
    if busy:
        caption = str(st.session_state.get("composer_caption", "") or "")
        uploaded = st.session_state.get("composer_image", uploaded)
        if uploaded is None or _text_len(caption) < 3:
            st.session_state["composer_busy"] = False
            st.warning("برای ارسال، یک تصویر و حداقل ۳ حرف متن لازم است.")
            return
        try:
            image = Image.open(uploaded).convert("RGB")
            st.session_state["composer_result"] = _explain_to_result(
                pipeline,
                caption=caption.strip(),
                image=image,
                gold_label=None,
                post_id=None,
                show_attention=False,
            )
        finally:
            st.session_state["composer_busy"] = False
        st.rerun()

    # Show last answer with unlocked «ارسال» button.
    if st.session_state.get("composer_result") is not None and uploaded is not None:
        _render_result(
            st.session_state["composer_result"],
            image=Image.open(uploaded).convert("RGB"),
            pipeline=None,
        )

    if not clicked:
        return

    caption = str(st.session_state.get("composer_caption", caption) or "")
    uploaded = st.session_state.get("composer_image", uploaded)

    if uploaded is None:
        st.warning("لطفاً یک تصویر بارگذاری کنید.")
        return
    if _text_len(caption) < 3:
        st.warning("متن باید حداقل ۳ حرف داشته باشد.")
        return

    st.session_state["composer_busy"] = True
    st.session_state.pop("composer_result", None)
    st.rerun()


def _explain_to_result(
    pipeline: Pipeline,
    *,
    caption: str,
    image,
    gold_label: str | None = None,
    post_id: str | None = None,
    show_attention: bool = False,
) -> dict:
    prediction, _feats, T_hat = pipeline.explain(caption, image)
    return {
        "caption": caption,
        "label": prediction.label,
        "confidence": float(prediction.confidence),
        "features": {k: float(v) for k, v in prediction.discrepancy_vector.items()},
        "T_hat": T_hat,
        "low_fidelity": bool(prediction.low_fidelity),
        "gold_label": gold_label,
        "post_id": post_id,
        "show_attention": show_attention,
    }


def _render_result(
    result: dict,
    *,
    image,
    pipeline: Pipeline | None = None,
) -> None:
    caption = str(result.get("caption") or "")
    label = str(result.get("label") or "")
    confidence = float(result.get("confidence") or 0.0)
    features = result.get("features") or {}
    T_hat = str(result.get("T_hat") or "")
    gold_label = result.get("gold_label")
    post_id = result.get("post_id")
    show_attention = bool(result.get("show_attention"))

    st.markdown(
        f'<div class="g-user"><p class="body">{html.escape(caption)}</p></div>',
        unsafe_allow_html=True,
    )
    st.image(image, use_container_width=True)

    panel_bits = [
        _prediction_prose(
            label,
            confidence,
            polarity_T=float(features.get("polarity_T")) if "polarity_T" in features else None,
            polarity_T_hat=(
                float(features.get("polarity_T_hat")) if "polarity_T_hat" in features else None
            ),
        )
    ]
    if gold_label is not None:
        gold_fa = LABEL_FA.get(gold_label, gold_label)
        if label == gold_label:
            panel_bits.append(
                f'<div class="g-ok">✓ با برچسب واقعی («{html.escape(gold_fa)}») مطابقت دارد.</div>'
            )
        else:
            panel_bits.append(
                f'<div class="g-warn">برچسب واقعی: «{html.escape(gold_fa)}» — '
                f"با پیش‌بینی عامل متفاوت است.</div>"
            )
    panel_bits.append(_metric_cards(features))
    panel_bits.append(
        f'<div class="g-note"><div class="lbl">توصیف تولیدشده (T̂ · SmolVLM)</div>'
        f'<p class="en">{html.escape(T_hat)}</p></div>'
    )
    if result.get("low_fidelity"):
        panel_bits.append(
            '<div class="g-warn">وفاداری تصویر–متن پایین است (Fvt). '
            "این پیش‌بینی را با احتیاط تفسیر کنید.</div>"
        )
    st.markdown(
        f'<div class="g-bot-panel">{"".join(panel_bits)}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("جزئیات شاخص‌های GDRM", expanded=False):
        rows = []
        for key in FEATURE_NAMES:
            rows.append(
                f"| {FEATURE_FA.get(key, key)} | `{key}` | "
                f"{float(features.get(key, 0.0)):.4f} | {FEATURE_HINT.get(key, '')} |"
            )
        st.markdown(
            "| شاخص | نماد | مقدار | توضیح |\n| --- | --- | --- | --- |\n" + "\n".join(rows)
        )

    if show_attention and post_id and pipeline is not None:
        with st.expander("نقشهٔ توجه متنی و تصویری", expanded=True):
            st.markdown(
                '<p class="g-section">توجه متنی (ParsBERT)</p>',
                unsafe_allow_html=True,
            )
            tokens, scores = attention_from_text(pipeline.bundle, caption)
            heatmap = render_text_heatmap(tokens, scores, title=None)
            st.markdown(f'<div class="g-heatmap">{heatmap}</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="g-section">توجه بصری (CLIP)</p>',
                unsafe_allow_html=True,
            )
            EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
            grid = attention_from_image(pipeline.bundle, image)
            path = overlay(image, grid, EXPLAIN_DIR / f"{post_id}.png")
            st.image(str(path), use_container_width=True)


def _render_conversation(
    pipeline: Pipeline,
    *,
    caption: str,
    image,
    gold_label: str | None = None,
    post_id: str | None = None,
    show_attention: bool = False,
) -> None:
    result = _explain_to_result(
        pipeline,
        caption=caption,
        image=image,
        gold_label=gold_label,
        post_id=post_id,
        show_attention=show_attention,
    )
    _render_result(result, image=image, pipeline=pipeline)
