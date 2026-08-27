# Pipeline status

## Proposal claims (`reports/REPORT.md`)

| Claim | Status |
| --- | --- |
| GDCNet-FA (Dsem/Dsen/Fvt) | **PASS** |
| Training-free inference | **PASS** |
| Binary sarcasm ≥ 70% (Dsem rule) | **PASS** — **85.9%** (also report binary F1 ~0.46; sarcasm is ~12%) |
| Multimodal +10 pp over unimodal | **PASS** — **+46.8 pp** sarcasm F1 |
| Peak VRAM ≤ 1 GiB (staged) | **PASS** — **0.94 GiB** |

## Key metrics (CLIP image polarity + retag)

- 5-class **macro-F1: 0.486 ± 0.017** (use this)
- 5-class accuracy: 0.484 ± 0.026 vs majority dummy **0.590** (dummy macro-F1 **0.148**)
- Sarcasm-subtype F1: **0.52** (upper bound until 137 gold sarcasm posts are hand-reviewed)
- Binary Dsem accuracy: **85.9%**; binary LogReg F1: **0.46**
- Staged peak VRAM: **0.945 GiB**
- Evaluated n = 1187 (positive 700, negative 243, neutral 107, +sarc 86, −sarc 51)

Speaker notes: `reports/DEFENSE.md`

## Honest implementation notes

- `polarity_T_hat` is CLIP smile/sad, not SmolVLM-caption polarity. `T̂` still feeds Dsem/Fvt.
- Text polarity: XLM-R social-media head + negation/contrast + a short list of funeral/irony **formulas** (not a sentiment dictionary).
- Taarof and cultural irony without text–image clash are out of the 5-class GDRM contract.

## Commands

```bash
python tasks.py dashboard
python tasks.py eval
```
