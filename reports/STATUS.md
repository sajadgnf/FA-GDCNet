# Pipeline status

Numbers from `reports/REPORT.md` and `reports/origin_split.json`. Prefer those files.

Eval n = **1351** (28 gold rows have missing images). Organic labeling queue is finished (55 recorded).

Clash is magnitude-gated: zero unless `|polarity_T| ≥ 0.20` and `|polarity_T_hat| ≥ 0.15`. Thin/spam captions have text polarity zeroed. The clash cut is CV-tuned for **accuracy** (PDF §6.3(2)), not F1.

## Proposal hypotheses

| Item | Status |
| --- | --- |
| H1 memory < 1 GiB (staged peak) | **YES** — 0.945 GiB staged. Dashboard **4.176 GiB**. |
| H2 sarcasm accuracy > 70% | **YES** as detection on mixed. Clash acc **78.8%**, dummy **76.4%**, P/R/F1 **0.59 / 0.36 / 0.44**. Organic also **YES** vs dummy (acc **87.0%**, dummy **86.6%**, P **0.66**). Organic recall is **0.11** (accuracy cut, not F1 cut). |
| H3 vs heavy model | **PASS** on CPU timing of SmolVLM-Instruct 2.2B, n=40. Ours 0.94 GiB / 1.94 s vs heavy 5.06 GiB CUDA footprint / 56.5 s CPU; 5-class acc 0.55 vs 0.30 (drop **−0.25**). CUDA generate was **1.10 s** but uses 5 GiB. |
| RQ2 sarcasm F1 ≥10 pp vs unimodal | Mixed **YES** (+22.8 pp). Organic-only **YES** (+13.8 pp). Absolute organic sarcasm-class F1 is **0.29**. |
| §8.3 Dsem/Dsen | Full sarcasm-F1 0.47 vs aux 0.43 vs no_clip 0.35. Main lever is CLIP smile/sad. |
| IAA | Kappa **0.778** on **230** overlap rows only. |

## Organic vs crafted (retrain)

| split | n | sarcasm | dummy acc | clash P / R / F1 | clash acc | sarcasm-class F1 | RQ2 Δ | beats dummy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 1351 | 319 | 0.764 | 0.595 / 0.364 / 0.438 | 0.788 | 0.465 | +0.228 | YES |
| organic | 1139 | 153 | 0.866 | 0.656 / 0.111 / 0.186 | 0.870 | 0.286 | +0.138 | YES |
| crafted | 212 | 166 | 0.217 | 0.785 / 0.970 / 0.868 | 0.769 | 0.627 | +0.182 | YES |

Organic clash precision is **0.66**, so accuracy beats the always-not-sarcasm dummy. Organic recall is low because the cut is tuned for accuracy, not F1. Organic is not a PDF hypothesis; mixed-set H2 is.

## Commands

```bash
python tasks.py train --from-cache
python -m eval.origin_split
python tasks.py eval
```
