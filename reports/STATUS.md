# Pipeline status (auto-generated)

## Completed

- Fixed M-CLIP loader (`XLM-Roberta-Large-Vit-B-32` + fp16 + load-time guards)
- Staged extraction with per-record checkpointing (`python tasks.py extract`)
- Split M-CLIP into text (CPU) + image (CUDA) sub-stages for the ≤1 GiB VRAM budget
- Added 144 weak-labeled sarcasm posts from archive pool (1187 total labeled)
- Full feature extraction on 1187 samples with live GDRM signals
- Train + eval suite (`python tasks.py finish`)

## Proposal claims (`reports/REPORT.md`)

| Claim | Status |
| --- | --- |
| GDCNet-FA (Dsem/Dsen/Fvt) | **PASS** — ablation shows Dsem drives performance |
| Training-free inference | **PASS** |
| Binary sarcasm ≥ 70% | **PASS** — Dsem CV-tuned rule: **72.4%** |
| Multimodal +10 pp over unimodal | **PASS** — **+17.8 pp** sarcasm F1 |
| Peak VRAM ≤ 1 GiB (staged) | **PASS** — **0.94 GiB** (captions); mCLIP text on CPU because XLM-R Large fp16 weights alone are ~1.04 GiB |

## Key metrics

- 5-class macro-F1: **0.307** (was 0.235 with broken M-CLIP)
- Sarcasm F1 multimodal vs baseline: **0.231 vs 0.052**
- Staged peak VRAM: **0.945 GiB** (SmolVLM captions)
- Staged latency: ~3.5 s/sample (caption-dominated)

## Commands

```bash
python tasks.py finish          # re-train + re-eval
python tasks.py dashboard       # explainability UI
python tasks.py augment-sarcasm # add more archive sarcasm posts
```

## Artifacts

- `artifacts/features.npz` — 1187 × 6 GDRM features
- `artifacts/clf.joblib` — 5-class classifier
- `artifacts/stages/*.jsonl` — resumable stage checkpoints
- `models/M-CLIP--XLM-Roberta-Large-Vit-B-32/` — local M-CLIP weights
