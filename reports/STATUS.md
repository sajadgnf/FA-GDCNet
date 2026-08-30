# Pipeline status

Bars are PDF §6.3. n_eval = 1043 after dropping 141 `weak-sarcasm-bootstrap` rows.

## Proposal hypotheses

| Item | Status |
| --- | --- |
| H1 memory < 1 GiB (staged peak) | **YES** — 0.945 GiB. Dashboard resident **4.176 GiB**. |
| H2 sarcasm accuracy > 70% | **YES** on the written bar — Dsem **85.4%**. Always-not-sarcasm dummy **88.3%**. Binary F1 **0.46**. |
| H3 vs heavy model | **NOT_RUN**. Qwen2-VL-2B crashed on load (0xC0000005). Not PASS. Use `python tasks.py eval --heavy` only on other hardware. |
| RQ2 sarcasm F1 ≥10 pp vs unimodal | **YES** on the **current** gold: multimodal **0.514** vs text polarity **0.250** (**+26.4 pp**). Still not independent of CLIP in features. |
| §8.3 Dsem/Dsen improve 5-class | **Not shown.** `aux_only` macro-F1 **0.487** > full six **0.480**. |
| Circularity (`no_clip` = drop CLIP hat + Dsen) | sarcasm-F1 **0.258** vs `aux_only` **0.497**. Subtype F1 depends on CLIP. |

## 5-class (OOF)

- Accuracy **0.478** vs majority dummy **0.587**
- Macro-F1 **0.480** vs dummy **0.148**
- Unimodal (same rows) accuracy **0.291**, macro-F1 **0.239**

## Labeling

- jsonl 1186; eval **1043** (excluded 141 bootstrap)
- Current sarcasm in jsonl 133; eval sarcasm 122 (11 were bootstrap)
- Old `relabel` Enter-confirm is **not** blind gold. Pending blind sarcasm: **133**
- Overlap list: `datasets/iaa_overlap_ids.txt` (133 sarcasm + 100 non-sarcasm)
- Kappa undefined until `datasets/iaa_second.jsonl` exists (`python tasks.py iaa`)

## Commands

```bash
python tasks.py relabel --only sarcasm
python tasks.py relabel --only all --ids-file datasets/iaa_overlap_ids.txt --out datasets/iaa_second.jsonl --annotator PERSON2
python tasks.py iaa
python tasks.py eval
# H3 only (unsafe on this GPU):
python tasks.py eval --heavy
```
