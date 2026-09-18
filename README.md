
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

`polarity_T_hat` is CLIP smile-vs-sad on the photo (not SmolVLM-caption polarity). `T̂` still feeds `Dsem` and `Fvt`.

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

| Command                                            | Description                                                        |
| -------------------------------------------------- | ------------------------------------------------------------------ |
| `python scripts/fetch_mclip.py`                    | Download M-CLIP weights into `models/` (required once).            |
| `python tasks.py extract`                          | Staged feature extraction (resumable; one backbone at a time).     |
| `python tasks.py train`                            | Train the sklearn classifier on the labeled dataset.               |
| `python tasks.py eval`                             | Metrics + sarcasm + staged profile + ablation + baseline + report. |
| `python tasks.py finish`                           | Extract → train (from cache) → full eval suite.                    |
| `python tasks.py dashboard`                        | Launch the Streamlit explainability dashboard.                     |
| `python tasks.py scrape --following --max-count N` | Scrape recent posts from accounts you follow.                      |
| `python tasks.py scrape --profile USER`            | Scrape a specific account.                                         |
| `python tasks.py scrape-session --user USER`       | Import Instagram session from browser cookies.                     |
| `python tasks.py enqueue-sarcasm`                   | Queue scrape-pool posts for blind sarcasm review (no auto gold). |
| `python tasks.py relabel --only candidates`         | Blind-review the candidate queue (label hidden; 1–5 required). |
| `python tasks.py relabel --export-overlap`          | Write sarcasm + 100 non-sarcasm ids for a second annotator. |
| `python tasks.py iaa`                               | Kappa from gold `blind-relabel` vs `datasets/iaa_second.jsonl`. |
| `python tasks.py label`                             | Rebuild labels from the raw pool (do **not** run on the defense dataset). |
| `python tasks.py augment-sarcasm`                   | Deprecated alias of `enqueue-sarcasm`. |
