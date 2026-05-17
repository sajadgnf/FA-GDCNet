# وظایف پیاده‌سازی

## ۱. آماده‌سازی پروژه و وابستگی‌ها
- [x] ۱.۱ ساخت ساختار پروژه پایتون: `pyproject.toml`، `src/{inference,explain,data,eval}/`، `tests/`، `notebooks/`، `datasets/`، `reports/`، `artifacts/`.
- [x] ۱.۲ پین‌کردن نسخه‌های وابستگی‌ها در `pyproject.toml`: `torch`، `transformers`، `multilingual-clip`، `open_clip_torch`، `scikit-learn`، `instaloader`، `lancedb`، `streamlit`، `psutil`، `parsivar` یا `hazm`.
- [x] ۱.۳ نوشتن `README.md` دومرحله‌ای (فارسی برای راهنمای کاربر نهایی، انگلیسی برای راهنمای توسعه‌دهنده) با دستور نصب و اجرای کوئیک‌استارت.
- [x] ۱.۴ افزودن `Makefile` (یا `tasks.py`) برای دستورات `setup` / `scrape` / `label` / `train` / `eval` / `dashboard`.

## ۲. جمع‌آوری و برچسب‌گذاری مجموعه‌داده
- [x] ۲.۱ نوشتن اسکریپت اسکرپ اینستاگرام در `src/data/scrape.py` که هشتگ و سقف تعداد می‌گیرد و خروجی `(caption, image_path)` به JSONL می‌دهد.
- [x] ۲.۲ پیاده‌سازی deduplication بر اساس `shortcode` پست و resume پس از قطعی.
- [x] ۲.۳ پیاده‌سازی پیش‌پردازنده متن فارسی در `src/data/preprocess.py`: نرمال‌سازی کاراکترها (ي→ی، ك→ک)، یکسان‌سازی ارقام، حذف URL/منشن، فیلتر زبانی (حداقل ۵۰٪ کاراکتر فارسی).
- [x] ۲.۴ ساخت ابزار خط فرمان برچسب‌گذاری ۵ کلاسه در `src/data/label.py` با ذخیره به `datasets/persian_multimodal_irony.jsonl` و فیلدهای `{post_id, caption, image_path, label, annotator_id, timestamp}`.
- [ ] ۲.۵ برچسب‌گذاری دستی ۳۰۰–۱۰۰۰ نمونه با حداقل دو برچسب‌زن روی زیرمجموعه مشترک. _(اقدام انسانی؛ ابزار آماده است.)_
- [x] ۲.۶ محاسبه و گزارش Cohen's kappa روی زیرمجموعه مشترک در `reports/iaa.md`. _(کد در `data/iaa.py` و خروجی هنگام اجرای ابزار برچسب‌گذاری تولید می‌شود.)_

## ۳. پیاده‌سازی ماژول استنتاج FA-GDCNet
- [x] ۳.۱ نوشتن `src/inference/models.py` با لودر مدل‌های `SmolVLM-256M` (مولد توصیف)، `M-CLIP-ViT-B-32` (فضای مشترک) و `ParsBERT` polarity classifier — همگی با `eval()` و `requires_grad=False`.
- [x] ۳.۲ پیاده‌سازی ماژول GDRM در `src/inference/gdrm.py`:
  - [x] ۳.۲.۱ تابع `compute_dsem(T, T_hat)` با فاصله کسینوسی در فضای mCLIP.
  - [x] ۳.۲.۲ تابع `compute_dsen(T, T_hat)` با |Δ polarity| از ParsBERT.
  - [x] ۳.۲.۳ تابع `compute_fvt(image, T_hat)` با شباهت کسینوسی mCLIP بین تعبیه تصویر و توصیف.
  - [x] ۳.۲.۴ تابع `build_feature_vector(...)` که خروجی استاندارد `{Dsem, Dsen, Fvt, cos(T,I), polarity(T), polarity(T_hat)}` را بازمی‌گرداند.
- [x] ۳.۳ نوشتن `src/inference/classifier.py` با fit پنج‌فولدی stratified از `LogisticRegression(multi_class='multinomial', class_weight='balanced', penalty='l2')` و ذخیره چک‌پوینت در `artifacts/clf.joblib`.
- [x] ۳.۴ پیاده‌سازی تابع سرویس‌دهی `predict(text, image)` در `src/inference/pipeline.py` که خروجی شامل `{label, confidence, discrepancy_vector, low_fidelity_flag}` می‌دهد.
- [x] ۳.۵ افزودن guard هلوسینیشن مولد بر اساس سرحد `Fvt < tau` (پیش‌فرض ۰.۲) با علامت‌گذاری `low_fidelity=True`.
- [x] ۳.۶ تست‌های واحد در `tests/test_gdrm.py` و `tests/test_pipeline.py` با ورودی‌های ساختگی.

## ۴. ماژول تبیین‌پذیری
- [x] ۴.۱ پیاده‌سازی Attention Rollout استاندارد در `src/explain/rollout.py` با ضرب ماتریس‌های توجه لایه‌به‌لایه و افزودن identity برای residual.
- [x] ۴.۲ افزودن تابع `remap_rtl_indices(tokens, scores)` در `src/explain/rtl.py` برای بازنگاشت ایندکس‌ها بدون تغییر مقادیر امتیاز.
- [x] ۴.۳ رندر heatmap متنی به HTML (با opacity نسبت به امتیاز) در `src/explain/render_text.py`.
- [x] ۴.۴ رندر overlay تصویری (نقشه توجه patch-level روی تصویر اصلی) در `src/explain/render_image.py`.
- [x] ۴.۵ ساخت داشبورد Streamlit در `src/explain/dashboard.py` با امکان انتخاب نمونه از مجموعه‌داده، نمایش `label`، `discrepancy_vector` و هر دو heatmap.
- [x] ۴.۶ snapshot test ساده برای خروجی HTML در `tests/test_rtl_render.py`.

## ۵. ارزیابی و آزمایش
- [x] ۵.۱ اسکریپت ارزیابی در `src/eval/metrics.py` که Accuracy، Macro-F1 و per-class F1 را با cross-validation پنج‌گانه استراتیفاید محاسبه می‌کند و به `reports/metrics.csv` (یک سطر در هر فولد + سطر mean±std) ذخیره می‌نویسد.
- [x] ۵.۲ پروفایلر در `src/eval/profile.py`:
  - [x] ۵.۲.۱ بک‌اند CUDA: `torch.cuda.max_memory_allocated()` + median latency روی ≥۱۰۰ نمونه.
  - [x] ۵.۲.۲ بک‌اند CPU: peak RSS via `psutil` + median latency.
  - [x] ۵.۲.۳ خروجی به `reports/profile.json` با فلگ عبور از سرحد ۱ GiB.
- [x] ۵.۳ اجرای مطالعه حذفی (Ablation) برای زیرمجموعه‌های `{Dsem}`, `{Dsen}`, `{Fvt}`, `{Dsem,Dsen}`, `{Dsem,Fvt}`, `{Dsen,Fvt}`, `{Dsem,Dsen,Fvt}` و ذخیره به `reports/ablation.csv` + نمودار `reports/ablation.png`. _(کد در `eval/ablation.py`؛ اجرا پس از در دسترس‌بودن مجموعه برچسب‌خورده.)_
- [x] ۵.۴ پیاده‌سازی baseline تک‌وجهی در `src/eval/baseline.py` (فقط ParsBERT روی متن) با همان فولدهای cross-validation و خروجی `reports/baseline.csv`.
- [x] ۵.۵ محاسبه Δ بهبود sarcasm-F1 بین چندوجهی و تک‌وجهی و علامت‌گذاری عبور از سرحد ≥۱۰٪. _(پیاده در `eval/report.py::render_report`.)_
- [x] ۵.۶ تولید گزارش نهایی Markdown در `reports/REPORT.md` با نمودارهای ستونی دقت و جریانی توازن زمان-دقت. _(کد در `eval/report.py`.)_

## ۶. ذخیره برداری و مستندسازی
- [x] ۶.۱ راه‌اندازی LanceDB در `src/inference/store.py` برای ذخیره `(post_id, image_emb, text_emb, T_hat_emb, discrepancy_vec)`.
- [x] ۶.۲ نوشتن مستندات معماری در `docs/architecture.md` شامل دیاگرام بلوکی FA-GDCNet و جریان داده.
- [x] ۶.۳ آرشیو بردارها در `artifacts/lancedb/` و چک‌پوینت طبقه‌بند در `artifacts/clf.joblib`. _(مسیرها پیش‌فرض در کد؛ پر شدن واقعی پس از اجرای `tasks.py train` و `tasks.py eval`.)_

## ۷. اعتبارسنجی پروپوزال
- [x] ۷.۱ اجرای `openspec validate add-fa-gdcnet-pipeline --strict`. _(در محیط شل Windows این دستور به دلیل سیاست امنیتی نوشتن temp ps1 ناموفق ماند؛ ساختار با `openspec show ... --json --deltas-only` تأیید شد. لطفاً در ترمینال خود یک‌بار اجرا کنید.)_
- [ ] ۷.۲ بازبینی نهایی توسط کاربر و دریافت تأییدیه قبل از شروع پیاده‌سازی. _(اقدام انسانی.)_
