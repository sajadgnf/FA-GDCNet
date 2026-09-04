# Pipeline status

Numbers from `reports/REPORT.md` and `reports/origin_split.json`. Prefer those files.

## Proposal hypotheses

| Item | Status |
| --- | --- |
| H1 memory < 1 GiB (staged peak) | **YES** if “method” = staged extract — 0.945 GiB. Dashboard **4.176 GiB**. |
| H2 sarcasm accuracy > 70% | Letter **YES** (clash acc 79.5%). Beats always-not-sarcasm dummy **NO** (dummy 79.5%). Clash P/R/F1 **0.50 / 0.48 / 0.49**. |
| H3 vs heavy model | **NOT settled.** n=4 Qwen2-VL-2B smoke: ours faster and <1 GiB; accuracy drop **0.25** (FAIL if taken at face value). n=40/150 failed (page file). |
| RQ2 sarcasm F1 ≥10 pp vs unimodal | Full set **YES** (+24.5 pp). Organic-only **YES** (+11.3 pp). |
| §8.3 Dsem/Dsen | Full sarcasm-F1 0.47 vs aux 0.42 vs no_clip 0.29. Organic sarcasm-class F1 **0.25**. |
| IAA | Kappa **0.778** on **230** overlap rows only. |

## Organic vs crafted (retrain)

| split | n | sarcasm | clash F1 | sarcasm-class F1 | RQ2 Δ |
| --- | --- | --- | --- | --- | --- |
| all | 1299 | 266 | 0.49 | 0.47 | +0.245 |
| organic | 1087 | 100 | 0.33 | 0.25 | +0.113 |
| crafted | 212 | 166 | 0.87 | — | +0.176 |

Clash detection on organic captions is weaker (F1 0.33, dummy acc 0.91). Mixed-set H2/RQ2 numbers are inflated by recaptions.

## Commands

```bash
python tasks.py eval
python -m eval.origin_split
python tasks.py eval --heavy
```
