# FA-GDCNet Architecture

## Block diagram

```
                                       ┌──────────────────────────────┐
                                       │       Persian caption T      │
                                       └──────────────┬───────────────┘
                                                      │
                                                      ▼
                       ┌──────────────────────────────────────────────────────┐
                       │  M-CLIP-ViT-B-32 text encoder  (frozen, multilingual) │
                       └──────────────────────────────┬───────────────────────┘
                                                      │ mCLIP_text(T)
                                                      ▼
                                    ┌─────────────────────────────────────┐
                                    │           GDRM (Section 3)          │
                                    │                                     │
                                    │   Dsem = 1 - cos(T, T̂)              │
                                    │   Dsen = ||p(T) - p(T̂)||_1          │
                                    │   Fvt  = cos(I, T̂)                  │
                                    │   + cos(T, I), pol(T), pol(T̂)       │
                                    │                                     │
                                    │   → DiscrepancyFeatures ∈ R⁶        │
                                    └────────────────┬────────────────────┘
                                                     │
                                                     ▼
                                    ┌─────────────────────────────────────┐
                                    │ LogisticRegression (5-class head)   │
                                    │ + LinearSVC fallback (if F1 < 0.40) │
                                    │ → joblib at artifacts/clf.joblib    │
                                    └────────────────┬────────────────────┘
                                                     │
                                                     ▼
                                ┌──────────────────────────────────────┐
                                │     Prediction(                      │
                                │       label,                         │
                                │       confidence,                    │
                                │       discrepancy_vector,            │
                                │       low_fidelity = (Fvt < τ=0.2),  │
                                │     )                                │
                                └──────────────────────────────────────┘

  ┌──────────────────────────────┐
  │         input image I        │
  └──────────────┬───────────────┘
                 │
                 ▼
   ┌────────────────────────────┐
   │   SmolVLM-256M (frozen)    │── generated description T̂ (en) ─┐
   └──────────────┬─────────────┘                                  │
                  │                                                ▼
                  ▼                                  ┌──────────────────────────────┐
   ┌──────────────────────────────┐                  │ M-CLIP text encoder ─► T̂_emb │
   │ CLIP-ViT-B/32 image encoder  │                  └──────────────┬───────────────┘
   │      (frozen, vision tower)  │                                 │
   └──────────────┬───────────────┘                                 │
                  │ mCLIP_image(I)                                  │
                  └─────────────────────────┬───────────────────────┘
                                            ▼
                                    (feeds into GDRM above)

  Persian caption T also fed to:
   ┌────────────────────────────────┐
   │ ParsBERT polarity (frozen)     │── softmax(p_neg, p_pos)    ─► GDRM
   └────────────────────────────────┘
```

## Constraints (enforced in code)

- `inference/models.assert_frozen(...)` is called by `load_backbones` and runs
  through every backbone to fail loudly if any parameter slipped out of
  `requires_grad=False` or out of `eval()` mode.
- `eval/profile.py` records peak VRAM (CUDA) or peak RSS (CPU) and tags the
  result with `under_1gib_budget = peak_memory_bytes <= 1 GiB`.
- `eval/metrics.py` footer reports whether sarcasm-class accuracy crosses the
  70 % hypothesis floor.
- `eval/report.py` computes the multimodal minus unimodal sarcasm-F1 delta
  and tags the ≥10 pp improvement hypothesis.

## Module-level rationale

- **`data/`** — pure stdlib + `hazm` (lazy). Everything in this package is
  testable without torch installed. The scraper is the only module that
  reaches out over the network and it imports `instaloader` lazily so the
  rest of the package stays light.
- **`inference/`** — heavy ML modules. `gdrm.py` and `classifier.py` are pure
  numpy / sklearn and unit-testable. `models.py` and `pipeline.py` import
  torch at module level; tests inject `BackboneBundle`-shaped fakes.
- **`explain/`** — `rtl.py` and `render_text.py` are pure-Python (testable).
  `rollout.py` has a pure-numpy core (`rollout`, `token_scores_from_rollout`)
  with thin torch-dependent helpers around it.
- **`eval/`** — every script reuses the pre-computed feature cache so the
  heavy backbones are run **once** (during `tasks.py train`); the
  metrics / ablation / baseline / profile scripts then load `artifacts/*.npz`
  and finish in seconds on a laptop.

## Data flow during a single dashboard query

1. User selects `post_id` from the sidebar (cached dataset index).
2. `Pipeline.from_pretrained()` is fetched from the Streamlit resource cache;
   on first call it loads the three frozen backbones (one-time cost).
3. `features_for(text, image)` runs SmolVLM caption + M-CLIP × 3 + ParsBERT.
4. `predict_from_features(features)` runs the sklearn classifier in <1 ms.
5. `attention_from_text(...)` runs ParsBERT with `output_attentions=True`,
   computes the rollout, and renders the HTML heatmap (RTL-remapped).
6. `attention_from_image(...)` runs the CLIP vision tower with attentions,
   builds the patch grid, and `overlay(...)` writes the PNG to
   `reports/explain/<post_id>.png`.
7. The dashboard surfaces every artifact + the `low_fidelity` banner when
   `Fvt < τ`.

## Vector store

`inference/store.py` (LanceDB) is optional. It is used by the evaluation
scripts to slice features for the ablation study without re-running the
backbones. It stores one row per sample:

```python
{
    "post_id":          str,
    "label":            str,
    "image_emb":        list[float],   # M-CLIP image embedding
    "text_emb":         list[float],   # M-CLIP text embedding of T
    "T_hat_emb":        list[float],   # M-CLIP text embedding of T̂
    "discrepancy_vec":  list[float],   # the 6-feature GDRM output
}
```

## Where the spec scenarios are realised

| Spec scenario | Code |
| --- | --- |
| Inference on a coherent positive post | `inference/pipeline.py::Pipeline.predict` |
| Inference on a sarcastic post with contradicting modalities | same |
| VRAM budget compliance | `eval/profile.py` writes `under_1gib_budget` |
| Frozen backbone guarantee | `inference/models.py::assert_frozen` |
| Computing semantic discrepancy Dsem | `inference/gdrm.py::compute_dsem` |
| Computing sentiment discrepancy Dsen | `inference/gdrm.py::compute_dsen` |
| Computing visual-textual fidelity Fvt | `inference/gdrm.py::compute_fvt` |
| Producing the discrepancy feature vector | `inference/gdrm.py::build_feature_vector` |
| Fitting the classifier | `inference/classifier.py::train` |
| Reloading a trained classifier | `inference/classifier.py::load` |
| Fallback classifier | `train()` retries with `_build_svc` |
| Low-fidelity rejection flag | `Pipeline.predict_from_features` sets `low_fidelity` |
| Fidelity reporting | `eval/report.py` (extensible) |
| Attention propagation across layers | `explain/rollout.py::rollout` |
| RTL token remap before rendering | `explain/rtl.py::remap_rtl_indices` |
| Image-side attention map | `explain/rollout.py::attention_from_image` |
| Text heatmap rendering | `explain/render_text.py::render_text_heatmap` |
| Image overlay rendering | `explain/render_image.py::overlay` |
| Joint artifact retrieval | dashboard composes both | 
| Loading a sample for inspection | `explain/dashboard.py::main` |
| Low-fidelity warning surface | dashboard banner |
| Successful scrape under rate limits | `data/scrape.py::main` |
| Idempotent re-runs | `data/scrape.py::_existing_shortcodes` |
| TOS compliance recording | README + `.gitignore` keeps images local |
| Character normalization | `data/preprocess.py::normalize_persian` |
| Language filter | `data/preprocess.py::is_persian_enough` |
| Labeling a sample | `data/label.py::_prompt_one` + `_build_canonical` |
| Inter-annotator agreement | `data/iaa.py::pairwise_kappa` |
| Skip and undo | `data/label.py` handles `s` / `u` / `q` |
| Reading the dataset | `data/schema.py::iter_dataset` |
| Schema validation | `data/schema.py::validate_record` raises `DatasetSchemaError` |
| Producing the metrics report | `eval/metrics.py::write_csv` |
| Meeting the accuracy hypothesis | `metrics.py` footer rows |
| Profiling on CUDA backend | `eval/profile.py::_measure_cuda` |
| Profiling on CPU backend | `eval/profile.py::_measure_cpu` |
| VRAM hypothesis check | `under_1gib_budget` field |
| Single-signal and pairwise runs | `eval/ablation.py::run` |
| Reporting ablation results | `ablation.py::write_csv` + `write_png` |
| Baseline run | `eval/baseline.py::evaluate` |
| Improvement margin check | `eval/report.py` Δ calculation |
