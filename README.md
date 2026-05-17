# FA-GDCNet — Lightweight Persian Multimodal Sentiment & Sarcasm Detector

<div dir="rtl">

## فارسی — راهنمای کاربر نهایی

FA-GDCNet یک خط لوله سبک و **بدون نیاز به آموزش مجدد ترنسفورمرها** برای تحلیل احساسات چندوجهی فارسی است. ورودی این سامانه یک جفت متن فارسی و تصویر (مثلاً پست اینستاگرام) است و خروجی یکی از پنج برچسب زیر:

- `positive` — احساس مثبت
- `negative` — احساس منفی
- `neutral` — خنثی
- `positive_sarcasm` — کنایه مثبت‌نما
- `negative_sarcasm` — کنایه منفی‌نما

این سامانه با ترکیب سه مدل سبک (همگی فریزشده) کار می‌کند:

1. **`SmolVLM-256M`** — تولید توصیف عینی (`T̂`) از تصویر.
2. **`M-CLIP-ViT-B-32`** — تعبیه متن فارسی و تصویر در یک فضای برداری مشترک.
3. **`ParsBERT`** — استخراج قطبیت احساسی متن فارسی.

سپس سه شاخص اختلاف محاسبه می‌شود:

- **Dsem**: فاصله کسینوسی بین متن کاربر `T` و توصیف تولیدشده `T̂` در فضای mCLIP.
- **Dsen**: تضاد قطبیت احساسی بین `T` و `T̂` با ParsBERT.
- **Fvt**: شباهت کسینوسی تصویر با توصیف تولیدشده برای کنترل توهم مدل.

این سه شاخص (به‌همراه چند ویژگی کمکی) به یک طبقه‌بند سبک sklearn (`LogisticRegression`) داده می‌شوند و برچسب نهایی تولید می‌شود.

### نصب سریع

```bash
git clone <repo>
cd test
python -m venv .venv
.venv\Scripts\activate          # ویندوز
pip install -e ".[dev]"
```

### کوئیک‌استارت

```bash
python tasks.py scrape --hashtag طبیعت --max-count 200
python tasks.py label
python tasks.py train
python tasks.py eval
python tasks.py dashboard
```

### نکته حقوقی

این مخزن **هیچ تصویری از اینستاگرام را بازنشر نمی‌کند**. اسکریپت اسکرپ صرفاً برای جمع‌آوری محلی داده برای پژوهش است. تنها بردارهای تعبیه و برچسب‌ها قابل اشتراک‌گذاری عمومی هستند.

</div>

---

## English — Developer Guide

FA-GDCNet is a lightweight, **training-free** multimodal pipeline for Persian sentiment and sarcasm detection. All transformer backbones are frozen; the only fitted parameters live in a small `scikit-learn` classifier (Logistic Regression by default, with a Linear SVM fallback).

### Architecture

```
caption (FA)  ─┐
               ├─► M-CLIP text emb ─┐
                                     │
image  ────────┼─► SmolVLM-256M caption (T̂) ─► M-CLIP text emb ─┐
               │                                                  │
               └─► M-CLIP image emb ──────────────────────────────┤
                                                                  ▼
                                                ┌──────────────────────────────┐
                                                │ GDRM: Dsem, Dsen, Fvt + aux │
                                                └──────────────────────────────┘
                                                                  │
                                                                  ▼
                                          ┌────────────────────────────────────┐
                                          │ LogisticRegression (5-class head)  │
                                          └────────────────────────────────────┘
                                                                  │
                                                                  ▼
                                          {label, confidence, low_fidelity_flag}
```

### Constraints

- VRAM ≤ 1 GiB at inference (`make profile` checks this).
- No backbone fine-tuning. Verified by `inference.models.assert_frozen(...)`.
- 5-class single label output.

### Repository Layout

```
src/
  data/        Instagram scraper, FA preprocessing, labeling tool, schema, kappa
  inference/   model loaders, GDRM, classifier, pipeline, LanceDB store
  explain/     Attention Rollout, RTL remap, HTML/PNG render, Streamlit dashboard
  eval/        metrics, profile, ablation, baseline, final report builder
tests/         Pure-Python unit tests (no heavy deps required)
docs/          Architecture and design notes
datasets/      Local scraped & labeled data (gitignored)
reports/       Generated CSV / JSON / PNG / Markdown outputs (gitignored)
artifacts/     Trained classifier + LanceDB vectors (gitignored)
notebooks/     Optional analysis notebooks
```

### Running Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests that exercise the heavy backbones (`tests/test_pipeline.py`, parts of `tests/test_gdrm.py`) inject lightweight fakes so they can run on a CPU-only machine without downloading model weights.

### CLI

A small `tasks.py` wraps the common workflows:

| Command | Description |
| --- | --- |
| `python tasks.py scrape --hashtag X --max-count N` | Scrape public Instagram posts under hashtag `X`. |
| `python tasks.py label` | Launch the CLI 5-class annotation tool. |
| `python tasks.py train` | Train the sklearn classifier on the labeled dataset. |
| `python tasks.py eval` | Run metrics + profile + ablation + baseline. |
| `python tasks.py dashboard` | Launch the Streamlit explainability dashboard. |

See `docs/architecture.md` for the full data flow and module-level rationale.

### License

MIT. Note that **scraped Instagram media must not be redistributed**; only embeddings and labels may be shared.
