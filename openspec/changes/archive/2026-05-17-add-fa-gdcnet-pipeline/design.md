# Design: FA-GDCNet Persian Multimodal Sentiment Pipeline

## Context
This is a green-field research codebase. The goal is a lightweight, **training-free** Persian multimodal sentiment / sarcasm detector that runs under a **1 GiB VRAM** budget on any backend (CPU or low-end GPU). It localizes the GDCNet idea (ICASSP 2026) for Persian by replacing the heavy backbones with `SmolVLM-256M`, `M-CLIP-ViT-B-32`, and `ParsBERT`, and by replacing end-to-end training with a small `scikit-learn` classifier fit on a hand-crafted discrepancy feature vector.

Stakeholders:
- Persian NLP researchers who need a reproducible baseline.
- Edge-deployment engineers who cannot afford >1 GiB VRAM or fine-tuning.
- Thesis committee evaluating computational efficiency vs. accuracy trade-off.

Constraints:
- Persian (Farsi) only; right-to-left text rendering for explanations.
- ≤ 1 GiB peak VRAM at inference time; any backend (CPU acceptable).
- **Training-free**: all transformer backbones frozen (`eval()`, `requires_grad=False`). The only fitted parameters are the sklearn classifier weights.
- Output: **5-class single label** `{positive, negative, neutral, positive_sarcasm, negative_sarcasm}`.
- Dataset: 300–1000 Instagram posts, manually labeled.

## Goals / Non-Goals

### Goals
- Reproducible inference pipeline loading SmolVLM-256M + M-CLIP + ParsBERT under the 1 GiB VRAM budget.
- Quantify text–image discrepancy via three signals (Dsem, Dsen, Fvt) and feed them to a sklearn classifier.
- Provide RTL-aware Attention Rollout for explainable predictions on Persian tokens.
- Establish a small Persian multimodal sarcasm benchmark (data collection + labeling + metrics).
- Demonstrate that the multimodal pipeline beats a text-only ParsBERT baseline on sarcasm F1 by at least 10 percentage points.

### Non-Goals
- Fine-tuning any transformer backbone (explicitly forbidden by the training-free constraint).
- Multilingual support beyond Persian. mCLIP is multilingual, but evaluation is Persian-only.
- Production-grade scalability, web service deployment, or distributed training. This is a research prototype.
- Beating large MLLM baselines on absolute accuracy. The goal is a favourable accuracy/cost trade-off, not state-of-the-art accuracy.

## Decisions

### Decision: Training-free discrepancy classifier on top of frozen backbones
- The pipeline computes three discrepancy signals: Dsem (cosine distance in mCLIP space between caption `T` and generated description `T̂`), Dsen (absolute difference of ParsBERT polarity probabilities for `T` vs `T̂`), and Fvt (cosine similarity in mCLIP between the image embedding and `T̂` embedding).
- They are concatenated with auxiliary features (`cos(T, I)`, `polarity(T)`, `polarity(T̂)`) into a vector `x ∈ R^k` (k ≈ 6–8) and fed to `sklearn.linear_model.LogisticRegression(multi_class='multinomial', class_weight='balanced', penalty='l2')` fit with 5-fold stratified cross-validation.
- **Alternatives considered:**
  - Fine-tuning ParsBERT or mCLIP — rejected: violates the training-free constraint and inflates VRAM beyond budget.
  - End-to-end MLP fusion head — rejected: more moving parts, less interpretable; sklearn LogReg is fast and inspectable.
  - Kernel SVM — kept as a fallback if LogReg under-performs on the small dataset.

### Decision: SmolVLM-256M as the captioner / hallucination probe
- SmolVLM-256M generates `T̂`, the "objective" description of the image. This caption is the bridge between the text and visual modalities for Dsem and Dsen.
- Per Hugging Face TB docs, SmolVLM-256M fits comfortably in <1 GB VRAM and supports CPU inference via `torch.compile` if needed.
- SmolVLM produces English captions. We feed both `T` (Persian) and `T̂` (English) directly into M-CLIP since M-CLIP is multilingual and shares one semantic space across languages. This avoids translation drift.
- **Alternatives considered:**
  - BLIP-2 — rejected: 2.7B+ params, blows the budget.
  - Florence-2 — rejected: heavier and less Persian-friendly downstream.
  - Translating `T̂` to Persian before mCLIP — kept as an optional path; will be reported as an ablation if time allows.

### Decision: M-CLIP-ViT-B-32 as the shared embedding space
- Multilingual-CLIP-ViT-B-32 provides aligned Persian text and image embeddings out of the box; ~150 M parameters fit the budget.
- **Alternatives considered:**
  - OpenCLIP ViT-B/16 — rejected: English only; would require a separate Persian text encoder, breaking the shared space.
  - Persian-tuned CLIP variants — rejected: not standardized or stable enough for a research baseline.

### Decision: 5-class single-label output
- Labels: `{positive, negative, neutral, positive_sarcasm, negative_sarcasm}`.
- Captures both polarity and sarcasm in one head, simplifying evaluation and metric reporting.
- **Alternatives considered:**
  - 3-class + sarcasm boolean (multi-task) — rejected: complicates loss handling and reporting on a small dataset.
  - 4-class flat with merged sarcasm — rejected: loses the polarity distinction inside sarcastic cases.

### Decision: RTL-aware Attention Rollout for explainability
- Standard Rollout multiplies attention matrices across transformer layers and adds the identity to model residuals, producing per-token attribution scores. We apply this to ParsBERT (caption side) and to the M-CLIP image encoder (image side).
- For Persian rendering, the **scores themselves are not changed**; we only remap the visualization index order so that the highest-attended token is rendered above the same word a Persian reader sees, preserving natural reading order.
- **Alternatives considered:**
  - GradCAM — rejected: not directly applicable to text transformers and harder to keep training-free.
  - Integrated Gradients — rejected: heavier per-sample compute; harder to fit a real-time dashboard.

### Decision: Instagram scraping via `instaloader` with rate-limit caching
- Scrape public posts under Persian hashtags. Persist `(post_id, caption, image_path)` to JSONL with shortcode-based deduplication and resume on rerun.
- Cache scrape results on disk to minimise re-fetch.
- **Alternatives considered:**
  - Twitter (X) — rejected: image-poor compared to Instagram; API is paywalled.
  - Telegram public channels — kept as a follow-up source if Instagram throughput is insufficient.
  - Manual download by annotators — used only to top up gaps; insufficient throughput on its own.

### Decision: LanceDB as the vector store
- Stores `(image_emb, text_emb, T̂_emb, discrepancy_vec)` per sample. Supports fast retrieval during evaluation and ablation.
- **Alternatives considered:**
  - FAISS — rejected: lacks metadata-aware queries we want for ablation slicing.
  - SQLite + JSON blobs — rejected: slower vector ops; less ergonomic for our access patterns.

### Decision: Streamlit dashboard for explainability
- Minimal effort to expose the inference output, discrepancy vector, and both heatmaps interactively.
- **Alternatives considered:**
  - Gradio — viable; chose Streamlit for richer layout primitives.
  - Custom React UI — rejected: out of scope and time-budget.

## Risks / Trade-offs

| Risk | Mitigation |
| --- | --- |
| SmolVLM English captions drift inside mCLIP's Persian space | Use mCLIP for both `T` and `T̂` (multilingual). Run a translate-then-embed ablation if needed. |
| Only 300–1000 labelled samples → LogReg overfits | Stratified 5-fold CV; L2 regularization; class-balanced weights; report mean ± std across folds. |
| Instagram TOS / scraping legality | Restrict to public posts; do not redistribute images; release only embeddings and labels. Document policy in `README.md`. |
| Generative hallucination in `T̂` | Apply Fvt threshold guard (`Fvt < τ` → `low_fidelity=True`); expose rejection rate as a fidelity metric in reports. |
| RTL Rollout visualization bugs | Snapshot tests on golden HTML; unit tests on `remap_rtl_indices`. |
| Memory-bandwidth-bound performance on small models | Apply `torch.compile` and TF32; document per-device profile in `reports/profile.json`. |
| Class imbalance (sarcasm is rare) | Stratified splits; `class_weight='balanced'`; oversample sarcasm classes during cross-validation if needed. |

## Migration Plan
Not applicable. The repository is empty; this is the first change introducing all modules from scratch. There is no prior data, model, or API to migrate.

## Open Questions
- Should `T̂` be translated to Persian before mCLIP embedding, or kept in English? Decided empirically as part of the ablation study.
- Will we release the dataset publicly? Only embeddings + labels can be released due to Instagram TOS; the final release policy is locked before archiving this change.
- What hashtag basket should the scraper target initially? To be locked in task ۲.۱ with input from the thesis advisor.
