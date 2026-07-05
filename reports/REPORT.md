# FA-GDCNet — Final Report

## Multimodal pipeline (5-fold CV)

| fold | accuracy | macro_f1 |
| --- | --- | --- |
| 1 | 0.4507042253521127 | 0.26285869186384236 |
| 2 | 0.2857142857142857 | 0.19509530018004595 |
| 3 | 0.38571428571428573 | 0.30385737069947594 |
| 4 | 0.4142857142857143 | 0.28338383838383835 |
| 5 | 0.38571428571428573 | 0.2877172817100222 |

## Unimodal ParsBERT baseline (same folds)

| fold | accuracy | macro_f1 |
| --- | --- | --- |
| 1 | 0.352112676056338 | 0.1854864433811802 |
| 2 | 0.34285714285714286 | 0.14817138707334784 |
| 3 | 0.42857142857142855 | 0.3324670997834416 |
| 4 | 0.4142857142857143 | 0.17264021887824899 |
| 5 | 0.4142857142857143 | 0.16821572002294893 |

## Sarcasm-F1 improvement check

- Multimodal sarcasm-F1 (macro of positive_sarcasm, negative_sarcasm): **0.1073**
- Unimodal baseline sarcasm-F1: **0.0737**
- Δ = **+0.0336** (+3.36 percentage points)
- Meets ≥10 pp hypothesis: **NO**

## Inference profile

- Backend: `cpu`
- Samples: `5`
- Median latency: `24853.5 ms`
- Peak memory: `3.896 GiB`
- under_1gib_budget: **NO**

## Ablation

![Ablation Macro-F1](ablation.png)

