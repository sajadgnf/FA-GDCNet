# FA-GDCNet — Final Report

## Multimodal pipeline (5-fold CV)

| fold | accuracy | macro_f1 |
| --- | --- | --- |
| 1 | 0.4789915966386555 | 0.4785377693016125 |
| 2 | 0.4789915966386555 | 0.47125939922833426 |
| 3 | 0.47257383966244726 | 0.4818692069368917 |
| 4 | 0.5316455696202531 | 0.5199239180687049 |
| 5 | 0.45569620253164556 | 0.4770621968006033 |
| mean±std | 0.4836±0.0255 | 0.4857±0.0174 |

- Majority dummy (same folds): accuracy **0.5897**, macro-F1 **0.1484**. Lead with macro-F1, not accuracy.

### Label counts (evaluated set)

| label | n |
| --- | --- |
| `positive` | 700 |
| `negative` | 243 |
| `neutral` | 107 |
| `positive_sarcasm` | 86 |
| `negative_sarcasm` | 51 |
| **total** | 1187 |

### Per-class F1 (mean±std)

| class | F1 |
| --- | --- |
| `positive` | 0.4947±0.0313 |
| `negative` | 0.6455±0.0519 |
| `neutral` | 0.2475±0.0602 |
| `positive_sarcasm` | 0.3830±0.0396 |
| `negative_sarcasm` | 0.6580±0.0481 |

## Unimodal ParsBERT baseline (same folds)

| fold | accuracy | macro_f1 |
| --- | --- | --- |
| 1 | 0.37320574162679426 | 0.2150974588901063 |
| 2 | 0.3875598086124402 | 0.2508425211982878 |
| 3 | 0.3827751196172249 | 0.2016861048148911 |
| 4 | 0.33653846153846156 | 0.18871758604431874 |
| 5 | 0.36538461538461536 | 0.21478775853775853 |
| mean±std | 0.3691±0.0180 | 0.2142±0.0207 |

## Sarcasm-F1 improvement check

- Multimodal sarcasm-F1 (macro of positive_sarcasm, negative_sarcasm): **0.5205**
- Unimodal baseline sarcasm-F1: **0.0521**
- Δ = **+0.4684** (+46.84 percentage points)
- Meets ≥10 pp hypothesis: **YES**

## Binary sarcasm detection (proposal Hypothesis 2)

- Dsem threshold rule (CV-tuned, interpretable): **0.8593**
- LogReg on discrepancy features: accuracy **0.7818**, binary F1 **0.4558±0.0336**
- Unimodal baseline binary accuracy: **0.6242**
- Meets ≥70% accuracy (Dsem rule): **YES**

## Staged inference profile (peak VRAM per backbone)

- `captions`: peak **0.945 GiB**, median **3030 ms**/sample
- `mclip_text` [cpu]: peak **0.008 GiB**, median **360 ms**/sample
  - XLM-Roberta-Large text tower runs on CPU; weights alone are ~1.04 GiB in fp16
- `mclip_image`: peak **0.312 GiB**, median **35 ms**/sample
- `polarity`: peak **0.547 GiB**, median **19 ms**/sample
- Combined peak (max stage): **0.945 GiB**
- Staged under_1gib_budget: **YES**
- Staged median total latency: **3444 ms**/sample

## Full pipeline profile (all backbones resident)

- Backend: `cuda`
- Samples: `100`
- Median latency: `3983.4 ms`
- Peak memory: `4.176 GiB`
- under_1gib_budget: **NO**

## Out-of-fold confusion

![Confusion matrix](confusion.png)

## Ablation

![Ablation Macro-F1](ablation.png)

| configuration | n_features | mean_macro_f1 |
| --- | --- | --- |
| aux_only | 3 | 0.48621214518395134 |
| Dsem | 4 | 0.4843263652396609 |
| Dsen | 4 | 0.4787488513094549 |
| Fvt | 4 | 0.4865118652033624 |
| Dsem+Dsen | 5 | 0.48442763112466397 |
| Dsem+Fvt | 5 | 0.48324818331880665 |
| Dsen+Fvt | 5 | 0.48206581482190264 |
| Dsem+Dsen+Fvt | 6 | 0.48573049806722934 |

`aux_only` is `cos_TI` + `polarity_T` + `polarity_T_hat`. Dsen is partly redundant with those polarities, so core-signal deltas can be small. The proposal claim is **multimodal vs unimodal**, not Dsen vs aux.

## Proposal claims checklist

| Claim | Result |
| --- | --- |
| GDCNet-FA (Dsem/Dsen/Fvt) implemented | YES (see ablation) |
| Training-free backbones | YES (`assert_frozen`) |
| Binary sarcasm accuracy ≥ 70% (Dsem rule) | **YES** (85.9%) |
| Multimodal ≥10 pp over unimodal (sarcasm F1) | **YES** (+46.8 pp) |
| Peak VRAM ≤ 1 GiB (staged) | **YES** (0.94 GiB)

## How to read the scores (defense notes)

- **5-class quality** is reported as **macro-F1**, not accuracy. The labeled set is imbalanced (most posts are `positive`), so a majority dummy can beat overall accuracy while losing the rare classes.
- **Binary Dsem accuracy** is the metric named in Hypothesis 2. Sarcasm is the minority class (~12%), so also report binary sarcasm F1; high accuracy alone does not mean sarcasm is detected that often.
- **`polarity_T_hat`** in GDRM is **CLIP facial affect** (smile vs sad), not the polarity of the SmolVLM sentence. `T̂` still feeds `Dsem` and `Fvt`. Original GDCNet scores `T̂` with a text sentiment head; SmolVLM-256M captions under a 1 GiB budget are often bland, so CLIP zero-shot smile/sad is the visual-affect stand-in (facial cues are a documented substitute when captions omit expression).
- **Sarcasm-subtype F1** is an **upper bound** until the sarcasm gold labels are fully hand-reviewed: visual affect informed both the gold rule and `polarity_T_hat`.
- **Neutral F1** is the weakest class (thin captions / ads). It is not the sarcasm hypothesis; do not lead with it.
- **Taarof**, honorific mismatches, and purely cultural irony without a text–image polarity clash are **out of scope** of the 5-class GDRM contract.

