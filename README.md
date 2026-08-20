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
2. **`M-CLIP` (`XLM-Roberta-Large-Vit-B-32`)** — تعبیه متن فارسی و تصویر در یک فضای برداری مشترک.
3. **`ParsBERT`** — استخراج قطبیت احساسی متن فارسی.

سپس سه شاخص اختلاف محاسبه می‌شود:

- **Dsem**: فاصله کسینوسی بین متن کاربر `T` و توصیف تولیدشده `T̂` در فضای mCLIP.
- **Dsen**: تضاد قطبیت احساسی بین `T` و `T̂` با ParsBERT.
- **Fvt**: شباهت کسینوسی تصویر با توصیف تولیدشده برای کنترل توهم مدل.

این سه شاخص (به‌همراه چند ویژگی کمکی) به یک طبقه‌بند سبک sklearn (`LogisticRegression`) داده می‌شوند و برچسب نهایی تولید می‌شود.

استخراج ویژگی‌ها **مرحله‌به‌مرحله** است (هر بار یک مدل روی GPU) تا اوج VRAM زیر ۱ گیگابایت بماند. برج متنی M-CLIP به‌خاطر حجم وزن‌ها روی **CPU** اجرا می‌شود.

### نصب سریع

```powershell
cd FA-GDCNet
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# PyTorch با CUDA (اگر GPU دارید):
# pip install torch --index-url https://download.pytorch.org/whl/cu124

# وزن‌های M-CLIP (~۲٫۱ گیگ — در گیت نیست):
python scripts/fetch_mclip.py
```

### کوئیک‌استارت

```powershell
# دمو مصنوعی (بدون اینستاگرام):
python scripts/proposal_demo.py

# خط لوله کامل روی دیتاست برچسب‌خورده (استخراج + آموزش + ارزیابی):
python tasks.py finish

# داشبورد توضیح‌پذیری:
python tasks.py dashboard
```

اگر فقط استخراج ویژگی را می‌خواهید (قابل ازسرگیری):

```powershell
python tasks.py extract
```

### اسکرپ اینستاگرام (الزام لاگین)

از سال ۲۰۲۴ به بعد، API عمومی اینستاگرام بدون ورود با خطای `403 login_required` پاسخ می‌دهد.

**روش پیشنهادی — پست‌های اکانت‌های واقعی (نه هشتگ):**

```powershell
python tasks.py scrape-session --user YOUR_IG_USERNAME --browser firefox
python tasks.py scrape --following --max-count 200 --session-user YOUR_IG_USERNAME
python tasks.py label
python tasks.py finish
python tasks.py dashboard
```

اگر `instaloader --login` شکست خورد، مسیر فایل session را صریح بدهید:

```powershell
python tasks.py scrape --following --max-count 200 `
  --session-user YOUR_IG_USERNAME `
  --session-file "C:\path\to\session-YOUR_IG_USERNAME"
```

**اگر `--following` خطای 400 داد یا لیست خالی بود:**

```powershell
copy datasets\raw\accounts.example.txt datasets\raw\accounts.txt
# accounts.txt را ویرایش کنید، سپس:
python tasks.py scrape --profiles-file datasets/raw/accounts.txt --max-count 200 --session-user YOUR_IG_USERNAME
```

اگر اسکرپ ممکن نیست، از `python scripts/proposal_demo.py` برای آزمایش بقیه مراحل استفاده کنید.

### نتایج ارزیابی

پس از `python tasks.py finish` (یا `eval`)، خلاصهٔ ادعاهای پروپوزال در `reports/REPORT.md` نوشته می‌شود.

### نکته حقوقی

این مخزن **هیچ تصویری از اینستاگرام را بازنشر نمی‌کند**. اسکریپت اسکرپ صرفاً برای جمع‌آوری محلی داده برای پژوهش است. تنها بردارهای تعبیه و برچسب‌ها قابل اشتراک‌گذاری عمومی هستند.

</div>

---

## English — Developer Guide

FA-GDCNet is a lightweight, **training-free** multimodal pipeline for Persian sentiment and sarcasm detection. All transformer backbones are frozen; the only fitted parameters live in a small `scikit-learn` classifier (Logistic Regression by default, with a Linear SVM fallback).

### Architecture

```
caption (FA)  ─┐
               ├─► M-CLIP text emb (CPU) ─┐
                                          │
image  ────────┼─► SmolVLM-256M caption (T̂) ─► M-CLIP text emb (CPU) ─┐
               │                                                        │
               └─► M-CLIP image emb (CUDA) ─────────────────────────────┤
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

Feature extraction is **staged** (one backbone resident at a time). The M-CLIP text tower runs on **CPU** because XLM-Roberta-Large fp16 weights alone are ~1.04 GiB; captions / image / polarity stages stay under the 1 GiB VRAM budget on CUDA.

### Constraints

- Peak **VRAM ≤ 1 GiB** in the staged path (`python -m eval.profile_staged` → `reports/profile_staged.json`).
- No backbone fine-tuning. Verified by `inference.models.assert_frozen(...)`.
- 5-class single label output.

### Setup

```powershell
cd FA-GDCNet
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
# Optional CUDA torch:
# pip install torch --index-url https://download.pytorch.org/whl/cu124
python scripts/fetch_mclip.py   # ~2.1 GB local M-CLIP checkpoint
```

### Quick start

```powershell
# Synthetic demo (no Instagram):
python scripts/proposal_demo.py

# Full thesis pipeline on the labeled dataset:
python tasks.py finish          # extract → train → eval → reports/REPORT.md
python tasks.py dashboard
```

Resume extraction only:

```powershell
python tasks.py extract
# or one stage at a time:
python tasks.py extract --stage captions
python tasks.py extract --stage mclip
python tasks.py extract --stage polarity
python tasks.py extract --stage assemble
```

### Repository Layout

```
src/
  data/        Instagram scraper, FA preprocessing, labeling tool, schema, kappa
  inference/   model loaders, GDRM, staged extraction, classifier, pipeline
  explain/     Attention Rollout, RTL remap, HTML/PNG render, Streamlit dashboard
  eval/        metrics, staged profile, ablation, baseline, final report builder
scripts/       fetch_mclip.py, proposal_demo.py, augment_sarcasm.py, …
tests/         Pure-Python unit tests (no heavy deps required)
docs/          Architecture and design notes
datasets/      Local scraped & labeled data (gitignored images)
reports/       Generated CSV / JSON / PNG / Markdown outputs
artifacts/     Features cache, stage JSONL checkpoints, trained classifier
models/        Local M-CLIP weights (gitignored; use scripts/fetch_mclip.py)
```

### Running Tests

```powershell
pip install -e ".[dev]"
pytest
```

Tests that exercise the heavy backbones (`tests/test_pipeline.py`, parts of `tests/test_gdrm.py`) inject lightweight fakes so they can run on a CPU-only machine without downloading model weights.

### CLI

`tasks.py` wraps the common workflows:

| Command | Description |
| --- | --- |
| `python scripts/fetch_mclip.py` | Download M-CLIP weights into `models/` (required once). |
| `python tasks.py extract` | Staged feature extraction (resumable; one backbone at a time). |
| `python tasks.py train` | Train the sklearn classifier on the labeled dataset. |
| `python tasks.py eval` | Metrics + sarcasm + staged profile + ablation + baseline + report. |
| `python tasks.py finish` | Extract → train (from cache) → full eval suite. |
| `python tasks.py dashboard` | Launch the Streamlit explainability dashboard. |
| `python tasks.py scrape --following --max-count N` | Scrape recent posts from accounts you follow. |
| `python tasks.py scrape --profile USER` | Scrape a specific account. |
| `python tasks.py scrape-session --user USER` | Import Instagram session from browser cookies. |
| `python tasks.py label` | CLI 5-class annotation tool. |
| `python tasks.py augment-sarcasm` | Append weak-labeled sarcasm posts from the archive pool. |

See `docs/architecture.md` for the full data flow and `reports/REPORT.md` for the latest proposal-claims checklist.

### License

MIT. Note that **scraped Instagram media must not be redistributed**; only embeddings and labels may be shared.
