# Inter-Annotator Agreement

## Per-pair Cohen's kappa

_No overlapping samples between two independent annotators._

`annotators` on a canonical jsonl row is a tag list (who touched the row), not two independent labels. Kappa is computed only from blind-relabel gold vs `--second` (`datasets/iaa_second.jsonl`).

The previous `relabel` tag (Enter confirming a machine label) is not IAA and is not a blind review.

Export overlap, then have a second person label the same ids into a separate file:

```
python tasks.py relabel --export-overlap
python tasks.py relabel --only all --ids-file datasets/iaa_overlap_ids.txt --annotator sjjd6502
python tasks.py relabel --only all --ids-file datasets/iaa_overlap_ids.txt --out datasets/iaa_second.jsonl --annotator PERSON2
python tasks.py iaa
```
