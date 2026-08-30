# FA-GDCNet — Final Report

## Multimodal pipeline (5-fold CV)

| fold | accuracy | macro_f1 |
| --- | --- | --- |
| 1 | 0.5454545454545454 | 0.541143981638827 |
| 2 | 0.47368421052631576 | 0.46773807657683586 |
| 3 | 0.4354066985645933 | 0.4523622489058467 |
| 4 | 0.44711538461538464 | 0.43712466843501324 |
| 5 | 0.49038461538461536 | 0.5035986223783173 |
| mean±std | 0.4784±0.0387 | 0.4804±0.0375 |

- Majority dummy (same folds): accuracy **0.5868**, macro-F1 **0.1479**. 5-class accuracy is not better than this dummy when the dummy is higher.

### Label counts (evaluated set)

| label | n |
| --- | --- |
| `positive` | 612 |
| `negative` | 213 |
| `neutral` | 96 |
| `positive_sarcasm` | 74 |
| `negative_sarcasm` | 48 |
| **total** | 1043 |

Excluded **141** `weak-sarcasm-bootstrap` rows from eval (unless later tagged `blind-relabel`).

### Per-class F1 (mean±std)

| class | F1 |
| --- | --- |
| `positive` | 0.4912±0.0535 |
| `negative` | 0.6286±0.0721 |
| `neutral` | 0.2545±0.0699 |
| `positive_sarcasm` | 0.3663±0.0484 |
| `negative_sarcasm` | 0.6614±0.0769 |

## Unimodal text-polarity baseline (same folds)

Head is `cardiffnlp/twitter-xlm-roberta-base-sentiment` (2-d), not ParsBERT. The proposal named ParsBERT; this is a deviation.

| fold | accuracy | macro_f1 |
| --- | --- | --- |
| 1 | 0.2966507177033493 | 0.2651482373765333 |
| 2 | 0.2822966507177033 | 0.22864391751850946 |
| 3 | 0.3349282296650718 | 0.25957548737882535 |
| 4 | 0.22596153846153846 | 0.19575963143890024 |
| 5 | 0.3173076923076923 | 0.2434307992202729 |
| mean±std | 0.2914±0.0373 | 0.2385±0.0249 |

## Research question 2 (not Hypothesis 3)

PDF §6.2 Q2: does Dsem+Dsen raise sarcasm detection by at least 10% versus a unimodal method? This is a research question. Hypothesis 3 in §6.3 is the heavy-model speed/accuracy comparison.

- Multimodal sarcasm-F1 (macro of positive_sarcasm, negative_sarcasm): **0.5139**
- Unimodal baseline sarcasm-F1: **0.2495**
- Δ = **+0.2644** (+26.44 percentage points)
- Meets ≥10 pp vs unimodal: **YES**

## Binary sarcasm detection (proposal Hypothesis 2)

- Dsem threshold rule (CV-tuned, interpretable): **0.8543**
- LogReg on discrepancy features: accuracy **0.7872**, binary F1 **0.4622±0.0580**
- Unimodal baseline binary accuracy: **0.5110**
- Always-not-sarcasm dummy accuracy: **0.8830** (H2 letter-pass does not imply beating this dummy).
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

| configuration | n_features | mean_macro_f1 | mean_sarcasm_f1 |
| --- | --- | --- | --- |
| aux_only | 3 | 0.486648141818148 | 0.4972781952967506 |
| no_clip | 4 | 0.2847899034926238 | 0.25848127895639944 |
| Dsem | 4 | 0.479608896097638 | 0.4975422131339659 |
| Dsen | 4 | 0.48008472083215875 | 0.5193008213135255 |
| Fvt | 4 | 0.4850700908403036 | 0.4967514174214214 |
| Dsem+Dsen | 5 | 0.47754718804567914 | 0.5138535064850853 |
| Dsem+Fvt | 5 | 0.48202534880633313 | 0.4989090141224102 |
| Dsen+Fvt | 5 | 0.4807727313204735 | 0.5193008213135255 |
| Dsem+Dsen+Fvt | 6 | 0.4803935195869681 | 0.5138535064850853 |

`aux_only` is `cos_TI` + `polarity_T` + `polarity_T_hat`. `no_clip` is `Dsem`+`Fvt`+`cos_TI`+`polarity_T` (drops CLIP `polarity_T_hat` and `Dsen`). PDF §8.3 required showing that Dsem and Dsen improve the final model. If `aux_only` matches or beats the full six-feature row, that contribution is **not shown**. If `no_clip` sarcasm-F1 collapses, subtype F1 depended on CLIP.

## Hypothesis 3 (heavy multimodal comparison)

- Model: `Qwen/Qwen2-VL-2B-Instruct`
- Ran: **NO**
- Reason: VLM worker exited with code 3221225477. Qwen2-VL-2B-Instruct crashed while loading shards on this host (Windows access violation 0xC0000005 on CUDA and CPU). Hypothesis 3 is NOT_RUN, not PASS.
- Samples: `0`
- Heavy 5-class accuracy: **None**
- Ours OOF accuracy (same ids): **None**
- Accuracy drop (heavy − ours): **None**
- Heavy median latency: **None** s
- Ours staged median latency: **3.443710950003151** s
- Heavy peak VRAM: **None** GiB
- Ours staged peak VRAM: **0.944849967956543** GiB
- Local VLM is larger than SmolVLM-256M but weaker than the Flamingo / Idefics-80B models named in the proposal literature review.
- Hypothesis 3: **NOT_RUN**

## Proposal hypotheses (PDF §6.3)

| Item | Result |
| --- | --- |
| H1 memory < 1 GiB (staged peak) | **YES** (0.94 GiB)
| H2 sarcasm accuracy > 70% (Dsem rule) | **YES** (85.4%; always-not-sarcasm dummy 0.8830) |
| H3 vs heavy model (<1 GiB, faster, drop <5%) | **NOT_RUN** |
| RQ2 multimodal sarcasm F1 ≥10 pp vs unimodal | **YES** (+26.4 pp) — research question, not H3 |
| Training-free backbones | YES (`assert_frozen`) |
| §8.3 Dsem/Dsen improve the model | see ablation (null if aux_only ≈ full) |

## How to read the scores (defense notes)

- **5-class quality** is reported as **macro-F1**, not accuracy. The labeled set is imbalanced (most posts are `positive`), so a majority dummy can beat overall accuracy while losing the rare classes.
- **Hypothesis 2** is sarcasm **accuracy > 70%**. The Dsem rule is the number stamped YES/NO against that bar. Always-not-sarcasm dummy accuracy and binary F1 must be read with it. Beating 70% is not the same as beating the dummy.
- **`polarity_T_hat`** in GDRM is **CLIP facial affect** (smile vs sad), not the polarity of the SmolVLM sentence. `T̂` still feeds `Dsem` and `Fvt`. Original GDCNet scores `T̂` with a text sentiment head; SmolVLM-256M captions under a 1 GiB budget are often bland, so CLIP zero-shot smile/sad is the visual-affect stand-in (facial cues are a documented substitute when captions omit expression).
- **Sarcasm-subtype F1** is still not independent of CLIP: `polarity_T_hat` in the feature vector is the same smile/sad channel that informed the original retag. One-human review of current sarcasm rows reduces but does not remove that overlap. Kappa is undefined without a second annotator.
- **Neutral F1** is the weakest class (thin captions / ads). It is not the sarcasm hypothesis; do not lead with it.
- **Taarof**, honorific mismatches, and purely cultural irony without a text–image polarity clash are **out of scope** of the 5-class GDRM contract.

