# FA-GDCNet — Final Report

## Multimodal pipeline (5-fold CV)

| fold | accuracy | macro_f1 |
| --- | --- | --- |
| 1 | 0.44649446494464945 | 0.43843401584721625 |
| 2 | 0.44814814814814813 | 0.4399400793162759 |
| 3 | 0.4444444444444444 | 0.43226936756435475 |
| 4 | 0.4925925925925926 | 0.48854223338022285 |
| 5 | 0.43703703703703706 | 0.4309955164083548 |
| mean±std | 0.4537±0.0198 | 0.4460±0.0215 |

- Majority dummy (same folds): accuracy **0.4856**, macro-F1 **0.1307**. 5-class accuracy is not better than this dummy when the dummy is higher.

### Label counts (evaluated set)

| label | n |
| --- | --- |
| `positive` | 656 |
| `negative` | 238 |
| `neutral` | 138 |
| `positive_sarcasm` | 189 |
| `negative_sarcasm` | 130 |
| **total** | 1351 |

### Per-class F1 (mean±std)

| class | F1 |
| --- | --- |
| `positive` | 0.4655±0.0167 |
| `negative` | 0.5430±0.0315 |
| `neutral` | 0.2908±0.0481 |
| `positive_sarcasm` | 0.4518±0.0255 |
| `negative_sarcasm` | 0.4791±0.0459 |

## Unimodal text-polarity baseline (same folds)

Head is `cardiffnlp/twitter-xlm-roberta-base-sentiment` (2-d), not ParsBERT. The proposal named ParsBERT; this is a deviation.

| fold | accuracy | macro_f1 |
| --- | --- | --- |
| 1 | 0.25830258302583026 | 0.19654273970304506 |
| 2 | 0.3037037037037037 | 0.2527522400862928 |
| 3 | 0.28888888888888886 | 0.2425244543720043 |
| 4 | 0.2740740740740741 | 0.25594827959234745 |
| 5 | 0.24814814814814815 | 0.21929491307630763 |
| mean±std | 0.2746±0.0201 | 0.2334±0.0225 |

## Research question 2 (not Hypothesis 3)

PDF §6.2 Q2: does Dsem+Dsen raise sarcasm detection by at least 10% versus a unimodal method? This is a research question. Hypothesis 3 in §6.3 is the heavy-model speed/accuracy comparison.

- Multimodal sarcasm-F1 (macro of positive_sarcasm, negative_sarcasm): **0.4655**
- Unimodal baseline sarcasm-F1: **0.2379**
- Δ = **+0.2275** (+22.75 percentage points)
- Meets ≥10 pp vs unimodal: **YES**

## Binary sarcasm detection (proposal Hypothesis 2)

- Clash rule (CV-tuned for accuracy, opposite text vs face polarity): accuracy **0.7876**, precision **0.5948**, recall **0.3635**, F1 **0.4383**
- Dsem accuracy cut (dummy-trap footnote, not the H2 detector): **0.7394**
- LogReg on discrepancy features: accuracy **0.6891**, binary F1 **0.4998±0.0315**
- Unimodal baseline binary accuracy: **0.5159**
- Always-not-sarcasm dummy accuracy: **0.7639** (sarcasm-class F1 of this dummy is 0).
- PDF §6.3(2) letter (clash accuracy ≥70%): **YES**
- Beats always-not-sarcasm accuracy: **YES**

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
| aux_only | 4 | 0.43675491183865656 | 0.4320549200205611 |
| no_clip | 4 | 0.3334717749719238 | 0.350927755897954 |
| Dsem | 5 | 0.4503906312911547 | 0.4434785235918969 |
| Dsen | 5 | 0.4283965378972738 | 0.4343097093114266 |
| Fvt | 5 | 0.4429483040951638 | 0.43867166878370967 |
| Dsem+Dsen | 6 | 0.4458482838097086 | 0.46678343420754037 |
| Dsem+Fvt | 6 | 0.4522982338860938 | 0.44612742802625727 |
| Dsen+Fvt | 6 | 0.4316424248683498 | 0.44225914768306024 |
| Dsem+Dsen+Fvt | 7 | 0.446036242503285 | 0.4654711060084889 |

`aux_only` is `cos_TI` + `polarity_T` + `polarity_T_hat` + `clash`. `no_clip` is `Dsem`+`Fvt`+`cos_TI`+`polarity_T` (drops CLIP `polarity_T_hat`, `Dsen`, and `clash`). PDF §8.3 required showing that Dsem and Dsen improve the final model. If `aux_only` matches or beats the full GDRM row, that contribution is **not shown**. If `no_clip` sarcasm-F1 collapses, subtype F1 depended on CLIP.

## Organic vs crafted captions (domain split)

Retrain 5-fold CV on each subset. `craft-*` ids are recaptioned faces, not Instagram caption–image pairs. OOF slice is mixed-train and can leak.

| split | n | sarcasm | clash P / R / F1 | clash acc | beats dummy acc | RQ2 Δ |
| --- | --- | --- | --- | --- | --- | --- |
| all | 1351 | 319 | 0.595 / 0.364 / 0.438 | 0.7876 | YES | +0.228 |
| organic | 1139 | 153 | 0.656 / 0.111 / 0.186 | 0.8701 | YES | +0.138 |
| crafted | 212 | 166 | 0.785 / 0.970 / 0.868 | 0.7690 | YES | +0.182 |

- Organic-only RQ2 Δ: **+0.1381** (meets ≥10 pp).

## Hypothesis 3 (heavy multimodal comparison)

- Model: `HuggingFaceTB/SmolVLM-Instruct`
- Ran: **YES**
- Samples: `40`
- Heavy 5-class accuracy: **0.3**
- Ours OOF accuracy (same ids): **0.55**
- Accuracy drop (heavy − ours): **-0.25000000000000006**
- Heavy median latency: **56.483344300009776** s
- Ours staged median latency: **1.9355206499994893** s
- Heavy peak VRAM: **5.063865661621094** GiB
- Ours staged peak VRAM: **0.944852352142334** GiB
- Latency timed on CPU (56.5 s/sample). CUDA characterization on this 6 GiB laptop: 1.10 s/sample at 5.06 GiB VRAM, which is outside the 1 GiB budget H3 tests. Local VLM is larger than SmolVLM-256M but weaker than the Flamingo / Idefics-80B models named in the proposal literature review.
- Hypothesis 3: **PARTIAL_LOCAL_STANDIN**
- The comparison partner is a **local stand-in**, not the heavy class named in the proposal (§3.4: Flamingo / BLIP-2 / Idefics). A run against it cannot settle H3, whatever the three components show.

## Proposal hypotheses (PDF §6.3)

| Item | Result |
| --- | --- |
| H1 memory < 1 GiB (staged peak) | **YES** (0.94 GiB)
| H2 sarcasm accuracy > 70% | **YES** (letter YES (78.8%); dummy 0.7639; beats dummy YES) |
| H3 vs heavy model (<1 GiB, faster, drop <5%) | **PARTIAL_LOCAL_STANDIN** |
| RQ2 multimodal sarcasm F1 ≥10 pp vs unimodal | **YES** (+22.8 pp) — research question, not H3 |
| Training-free backbones | YES (`assert_frozen`) |
| §8.3 Dsem/Dsen improve the model | see ablation (null if aux_only ≈ full) |

## Deviations from the proposal PDF

- **ParsBERT** is named for Dsen (§8.1, §9). The text polarity head is `cardiffnlp/twitter-xlm-roberta-base-sentiment`. Image polarity is CLIP smile/sad, not ParsBERT on SmolVLM `T̂`.
- **H1** is measured as staged extract (one backbone resident). The PDF does not define staging. Dashboard with all towers loaded exceeds 1 GiB.
- **§5** specifies Instagram text–image pairs. Crafted recaptions (`craft-*`) keep the photo and replace the caption.
- **§8.2 scenario 2** (implicit / common-knowledge sarcasm) is not evaluated.
- **H3** default local VLM is `HuggingFaceTB/SmolVLM-Instruct` (2.2B). `Qwen2-VL-2B-Instruct` crashed on this Windows host while loading shards.

## How to read the scores

- **5-class quality** is reported as **macro-F1**, not accuracy. The labeled set is imbalanced (most posts are `positive`), so a majority dummy can beat overall accuracy while losing the rare classes.
- **Hypothesis 2** in the PDF is sarcasm **accuracy > 70%**. That bar can be met by always predicting not-sarcasm when sarcasm is rare. Detection evidence is precision/recall/F1 and whether accuracy beats the always-not-sarcasm dummy. A previous F1≥0.40 conjunct was not in the PDF.
- **`clash`** is zero unless `|polarity_T| ≥ 0.20` and `|polarity_T_hat| ≥ 0.15` (same floors as the inference polarity-conflict rule). Weak smile/sad scores are not counted as a text–image clash. Hashtag-only and prompt-spam captions have text polarity zeroed so they cannot fire clash. The clash cut is CV-tuned for **accuracy** (PDF §6.3(2)).
- **`polarity_T_hat`** in GDRM is **CLIP facial affect** (smile vs sad), not the polarity of the SmolVLM sentence. `T̂` still feeds `Dsem` and `Fvt`. Original GDCNet scores `T̂` with a text sentiment head; SmolVLM-256M captions under a 1 GiB budget are often bland, so CLIP zero-shot smile/sad is the visual-affect stand-in (facial cues are a documented substitute when captions omit expression).
- **Sarcasm-subtype F1** can still use CLIP: `polarity_T_hat` is smile/sad in the feature vector. Kappa, if computed, is only on the overlap file vs gold `blind-relabel` rows (`reports/iaa.md`), not on the full eval set. Most non-sarcasm eval rows were not blindly relabeled.
- **Neutral F1** is the weakest class (thin captions / ads). It is not the sarcasm hypothesis; do not lead with it.
- **Taarof**, honorific mismatches, and purely cultural irony without a text–image polarity clash are **out of scope** of the 5-class GDRM contract.

