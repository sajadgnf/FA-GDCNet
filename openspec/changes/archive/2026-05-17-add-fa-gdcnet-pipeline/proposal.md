# Change: افزودن سامانه FA-GDCNet برای تحلیل احساسات چندوجهی فارسی

## Why
شبکه‌های اجتماعی تصویرمحور (به‌ویژه اینستاگرام) پر از پست‌هایی هستند که متن و تصویر در آن‌ها تضاد معنایی دارند (کنایه چندوجهی). روش‌های موجود یا تک‌وجهی‌اند و این تضاد را نمی‌بینند، یا چندوجهی‌اند ولی به مدل‌های سنگین چندمیلیارد پارامتری و آموزش گران‌قیمت نیاز دارند که برای زبان فارسی به دلیل محدودیت داده و منابع، کاربردپذیری عملی ندارند.

این پروپوزال یک خط لوله سبک، **بدون آموزش مجدد ترنسفورمرها** (Training-Free) و قابل اجرا با کمتر از ۱ گیگابایت VRAM ارائه می‌دهد که با ترکیب `SmolVLM-256M`، `M-CLIP-ViT-B-32` و طبقه‌بند قطبیت فارسی `ParsBERT`، سه شاخص تضاد (Dsem, Dsen, Fvt) را استخراج کرده و با یک طبقه‌بند سبک sklearn روی بردار اختلاف، کنایه چندوجهی فارسی را تشخیص می‌دهد.

## What Changes
- افزودن قابلیت `multimodal-sentiment`: خط لوله استنتاج FA-GDCNet شامل بارگذاری مدل‌های پایه (همه فریزشده)، ماژول بازنمایی اختلاف مولد (GDRM) با سه شاخص Dsem/Dsen/Fvt، طبقه‌بند سبک Logistic Regression / Linear SVM روی بردار اختلاف، و خروجی **۵ کلاسه**: `{positive, negative, neutral, positive_sarcasm, negative_sarcasm}`.
- افزودن قابلیت `attention-explainability`: الگوریتم اصلاح‌شده Attention Rollout برای زبان‌های راست‌به‌چپ به‌همراه ماژول رندر heatmap روی توکن‌های فارسی و نواحی تصویر و یک داشبورد ساده Streamlit برای بازرسی نمونه‌ها.
- افزودن قابلیت `persian-multimodal-dataset`: اسکریپت اسکرپ مستقیم اینستاگرام (با احترام به rate-limit)، پیش‌پردازنده متن فارسی، ابزار خط فرمان برچسب‌گذاری ۵ کلاسه با محاسبه Cohen's kappa و فرمت ذخیره‌سازی استاندارد JSONL.
- افزودن قابلیت `evaluation-benchmark`: چارچوب ارزیابی شامل Accuracy / Macro-F1 / per-class F1 با cross-validation پنج‌گانه، پروفایلینگ زمان و حافظه (CUDA و CPU)، مطالعه حذفی (Ablation) برای سهم Dsem/Dsen/Fvt، و مقایسه با خط پایه تک‌وجهی ParsBERT با چک سرحد بهبود ≥۱۰٪.

## Impact
- **Affected specs (همگی جدید):** `multimodal-sentiment`، `attention-explainability`، `persian-multimodal-dataset`، `evaluation-benchmark`.
- **Affected code:** مخزن کاملاً خالی است؛ کدبیس از صفر ساخته می‌شود. ساختار پیشنهادی:
  - `src/inference/` — لودر مدل‌ها، ماژول GDRM، طبقه‌بند سبک، تابع `predict()`.
  - `src/explain/` — Attention Rollout + remap RTL + رندر heatmap + داشبورد Streamlit.
  - `src/data/` — اسکرپر اینستاگرام، نرمال‌سازی متن، ابزار برچسب‌گذاری.
  - `src/eval/` — اسکریپت‌های متریک‌گیری، پروفایلینگ، ablation و baseline.
  - `datasets/`, `reports/`, `artifacts/` — خروجی‌ها و نقاط ذخیره.
- **External dependencies (new):** `torch`, `transformers`, `open_clip_torch`, `multilingual-clip`, `scikit-learn`, `instaloader` (یا معادل)، `lancedb`، `streamlit`، `psutil`، `parsivar`/`hazm`.
- **محدودیت‌های سخت:** پیک VRAM در زمان استنتاج ≤ ۱ GiB روی هر بک‌اند (CPU یا GPU). همه ترنسفورمرها در حالت `eval()` و `requires_grad=False`؛ تنها پارامترهای قابل برازش، وزن‌های طبقه‌بند خطی sklearn هستند.
- **محدودیت‌های نرم:** مجموعه‌داده ۳۰۰–۱۰۰۰ نمونه‌ای دستی برچسب‌خورده؛ به دلیل TOS اینستاگرام، فقط بردارهای تعبیه و برچسب‌ها می‌توانند منتشر شوند نه خود تصاویر.
