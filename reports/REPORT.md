# FA-GDCNet — Final Report

## Multimodal pipeline (5-fold CV)

| fold | accuracy | macro_f1 |
| --- | --- | --- |
| 1 | 0.47307692307692306 | 0.45073074391046736 |
| 2 | 0.46923076923076923 | 0.45015034418060945 |
| 3 | 0.5153846153846153 | 0.46695565130856903 |
| 4 | 0.4461538461538462 | 0.4246403415735226 |
| 5 | 0.5135135135135135 | 0.4790318666944703 |
| mean±std | 0.4835±0.0269 | 0.4543±0.0183 |

- Majority dummy (same folds): accuracy **0.5065**, macro-F1 **0.1345**. 5-class accuracy is not better than this dummy when the dummy is higher.

### Label counts (evaluated set)

| label | n |
| --- | --- |
| `positive` | 658 |
| `negative` | 237 |
| `neutral` | 138 |
| `positive_sarcasm` | 170 |
| `negative_sarcasm` | 96 |
| **total** | 1299 |

### Per-class F1 (mean±std)

| class | F1 |
| --- | --- |
| `positive` | 0.5257±0.0339 |
| `negative` | 0.5794±0.0489 |
| `neutral` | 0.2314±0.0496 |
| `positive_sarcasm` | 0.4527±0.0519 |
| `negative_sarcasm` | 0.4823±0.0565 |

## Unimodal text-polarity baseline (same folds)

Head is `cardiffnlp/twitter-xlm-roberta-base-sentiment` (2-d), not ParsBERT. The proposal named ParsBERT; this is a deviation.

| fold | accuracy | macro_f1 |
| --- | --- | --- |
| 1 | 0.2846153846153846 | 0.22250293181244088 |
| 2 | 0.3076923076923077 | 0.279725585149314 |
| 3 | 0.27692307692307694 | 0.2362002231722729 |
| 4 | 0.24615384615384617 | 0.22376014403239228 |
| 5 | 0.277992277992278 | 0.2218790831421716 |
| mean±std | 0.2787±0.0197 | 0.2368±0.0221 |

## Research question 2 (not Hypothesis 3)

PDF §6.2 Q2: does Dsem+Dsen raise sarcasm detection by at least 10% versus a unimodal method? This is a research question. Hypothesis 3 in §6.3 is the heavy-model speed/accuracy comparison.

- Multimodal sarcasm-F1 (macro of positive_sarcasm, negative_sarcasm): **0.4675**
- Unimodal baseline sarcasm-F1: **0.2221**
- Δ = **+0.2454** (+24.54 percentage points)
- Meets ≥10 pp vs unimodal: **YES**

## Binary sarcasm detection (proposal Hypothesis 2)

- Clash rule (CV-tuned for F1, opposite text vs face polarity): accuracy **0.7945**, precision **0.5006**, recall **0.4848**, F1 **0.4908**
- Dsem accuracy cut (dummy-trap footnote, not the H2 detector): **0.7706**
- LogReg on discrepancy features: accuracy **0.7113**, binary F1 **0.4972±0.0289**
- Unimodal baseline binary accuracy: **0.5058**
- Always-not-sarcasm dummy accuracy: **0.7952** (sarcasm-class F1 of this dummy is 0).
- PDF §6.3(2) letter (clash accuracy ≥70%): **YES**
- Beats always-not-sarcasm accuracy: **NO**

## Staged inference profile (peak VRAM per backbone)

- `captions`: peak **0.945 GiB**, median **1477 ms**/sample
- `mclip_text` [cpu]: peak **0.008 GiB**, median **397 ms**/sample
  - XLM-Roberta-Large text tower runs on CPU; weights alone are ~1.04 GiB in fp16
- `mclip_image`: peak **0.312 GiB**, median **36 ms**/sample
- `polarity`: peak **0.547 GiB**, median **26 ms**/sample
- Combined peak (max stage): **0.945 GiB**
- Staged under_1gib_budget: **YES**
- Staged median total latency: **1936 ms**/sample

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

| configuration | n_features | mean_macro_f1 | mean_sarcasm_f1 |
| --- | --- | --- | --- |
| aux_only | 4 | 0.4328575658612065 | 0.4192143360595484 |
| no_clip | 4 | 0.29678342472217734 | 0.2921721690346358 |
| Dsem | 5 | 0.44331861982440685 | 0.4421491371526643 |
| Dsen | 5 | 0.4264000717411715 | 0.4340652660770834 |
| Fvt | 5 | 0.4346929614824173 | 0.4241106076999263 |
| Dsem+Dsen | 6 | 0.45043400462391964 | 0.4640113455111491 |
| Dsem+Fvt | 6 | 0.4454677257755722 | 0.4429365915696968 |
| Dsen+Fvt | 6 | 0.4326226719510785 | 0.43991612749027365 |
| Dsem+Dsen+Fvt | 7 | 0.4543017895335278 | 0.4675076675718227 |

`aux_only` is `cos_TI` + `polarity_T` + `polarity_T_hat` + `clash`. `no_clip` is `Dsem`+`Fvt`+`cos_TI`+`polarity_T` (drops CLIP `polarity_T_hat`, `Dsen`, and `clash`). PDF §8.3 required showing that Dsem and Dsen improve the final model. If `aux_only` matches or beats the full GDRM row, that contribution is **not shown**. If `no_clip` sarcasm-F1 collapses, subtype F1 depended on CLIP.

## Organic vs crafted captions (domain split)

Retrain 5-fold CV on each subset. `craft-*` ids are recaptioned faces, not Instagram caption–image pairs. OOF slice is mixed-train and can leak.

| split | n | sarcasm | clash P / R / F1 | clash acc | beats dummy acc | RQ2 Δ |
| --- | --- | --- | --- | --- | --- | --- |
| all | 1299 | 266 | 0.501 / 0.485 / 0.491 | 0.7945 | NO | +0.245 |
| organic | 1087 | 100 | 0.320 / 0.360 / 0.334 | 0.8638 | NO | +0.113 |
| crafted | 212 | 166 | 0.785 / 0.970 / 0.868 | 0.7690 | YES | +0.176 |

- Organic-only RQ2 Δ: **+0.1134** (meets ≥10 pp).

## Hypothesis 3 (heavy multimodal comparison)

- Model: `Qwen/Qwen2-VL-2B-Instruct`
- Ran: **YES**
- Reason: Smoke test n=4 completed. Follow-up n=40 and n=150 failed to load Qwen2-VL-2B (Windows paging file OS error 1455). n=4 cannot settle Hypothesis 3.
- Samples: `4`
- Heavy 5-class accuracy: **0.75**
- Ours OOF accuracy (same ids): **0.5**
- Accuracy drop (heavy − ours): **0.25**
- Heavy median latency: **8.498694300000352** s
- Ours staged median latency: **1.9355206499994893** s
- Heavy peak VRAM: **5.559333324432373** GiB
- Ours staged peak VRAM: **0.944852352142334** GiB
- Local VLM is larger than SmolVLM-256M but weaker than the Flamingo / Idefics-80B models named in the proposal literature review. Accuracy on 4 stratified posts is not a hypothesis test.
- Hypothesis 3: **FAIL**
- This run does **not** settle H3: sample size is too small (or the load failed on a larger n).

## Proposal hypotheses (PDF §6.3)

| Item | Result |
| --- | --- |
| H1 memory < 1 GiB (staged peak) | **YES** (0.94 GiB)
| H2 sarcasm accuracy > 70% | **NO as detection** (letter YES (79.5%); dummy 0.7952; beats dummy NO) |
| H3 vs heavy model (<1 GiB, faster, drop <5%) | **FAIL** |
| RQ2 multimodal sarcasm F1 ≥10 pp vs unimodal | **YES** (+24.5 pp) — research question, not H3 |
| Training-free backbones | YES (`assert_frozen`) |
| §8.3 Dsem/Dsen improve the model | see ablation (null if aux_only ≈ full) |

## Deviations from the proposal PDF

- **ParsBERT** is named for Dsen (§8.1, §9). The text polarity head is `cardiffnlp/twitter-xlm-roberta-base-sentiment`. Image polarity is CLIP smile/sad, not ParsBERT on SmolVLM `T̂`.
- **H1** is measured as staged extract (one backbone resident). The PDF does not define staging. Dashboard with all towers loaded exceeds 1 GiB.
- **§5** specifies Instagram text–image pairs. Crafted recaptions (`craft-*`) keep the photo and replace the caption.
- **§8.2 scenario 2** (implicit / common-knowledge sarcasm) is not evaluated.

## How to read the scores

- **5-class quality** is reported as **macro-F1**, not accuracy. The labeled set is imbalanced (most posts are `positive`), so a majority dummy can beat overall accuracy while losing the rare classes.
- **Hypothesis 2** in the PDF is sarcasm **accuracy > 70%**. That bar can be met by always predicting not-sarcasm when sarcasm is rare. Detection evidence is precision/recall/F1 and whether accuracy beats the always-not-sarcasm dummy. A previous F1≥0.40 conjunct was not in the PDF.
- **`polarity_T_hat`** in GDRM is **CLIP facial affect** (smile vs sad), not the polarity of the SmolVLM sentence. `T̂` still feeds `Dsem` and `Fvt`. Original GDCNet scores `T̂` with a text sentiment head; SmolVLM-256M captions under a 1 GiB budget are often bland, so CLIP zero-shot smile/sad is the visual-affect stand-in (facial cues are a documented substitute when captions omit expression).
- **Sarcasm-subtype F1** can still use CLIP: `polarity_T_hat` is smile/sad in the feature vector. Kappa, if computed, is only on the overlap file vs gold `blind-relabel` rows (`reports/iaa.md`), not on the full eval set. Most non-sarcasm eval rows were not blindly relabeled.
- **Neutral F1** is the weakest class (thin captions / ads). It is not the sarcasm hypothesis; do not lead with it.
- **Taarof**, honorific mismatches, and purely cultural irony without a text–image polarity clash are **out of scope** of the 5-class GDRM contract.

