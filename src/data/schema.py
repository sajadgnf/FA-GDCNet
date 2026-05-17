"""Dataset schema for the Persian multimodal irony corpus.

The schema is intentionally documented in code (`DatasetRecord`) and enforced by
`validate_record` so the spec scenario "Schema validation" can fail loudly with
a `DatasetSchemaError` naming the offending line and field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

# Spec contract: exactly these five class strings, in this canonical order.
LABELS: tuple[str, ...] = (
    "positive",
    "negative",
    "neutral",
    "positive_sarcasm",
    "negative_sarcasm",
)

REQUIRED_FIELDS: tuple[str, ...] = (
    "post_id",
    "caption",
    "image_path",
    "label",
    "annotators",
)


class DatasetSchemaError(ValueError):
    """Raised when a dataset record is missing a required field or has an unknown label."""

    def __init__(self, message: str, *, line_no: int | None = None, field_name: str | None = None):
        self.line_no = line_no
        self.field_name = field_name
        prefix_parts = []
        if line_no is not None:
            prefix_parts.append(f"line {line_no}")
        if field_name is not None:
            prefix_parts.append(f"field {field_name!r}")
        prefix = f"[{', '.join(prefix_parts)}] " if prefix_parts else ""
        super().__init__(prefix + message)


@dataclass
class DatasetRecord:
    """One labeled multimodal sample."""

    post_id: str
    caption: str
    image_path: str
    label: str
    annotators: list[str] = field(default_factory=list)
    kappa: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, line_no: int | None = None) -> "DatasetRecord":
        validate_record(raw, line_no=line_no)
        return cls(
            post_id=str(raw["post_id"]),
            caption=str(raw["caption"]),
            image_path=str(raw["image_path"]),
            label=str(raw["label"]),
            annotators=list(raw.get("annotators") or []),
            kappa=float(raw["kappa"]) if raw.get("kappa") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id,
            "caption": self.caption,
            "image_path": self.image_path,
            "label": self.label,
            "annotators": list(self.annotators),
            "kappa": self.kappa,
        }


def validate_record(raw: Any, *, line_no: int | None = None) -> None:
    """Raise DatasetSchemaError if `raw` is not a valid record."""
    if not isinstance(raw, dict):
        raise DatasetSchemaError("record must be a JSON object", line_no=line_no)
    for key in REQUIRED_FIELDS:
        if key not in raw:
            raise DatasetSchemaError("missing required field", line_no=line_no, field_name=key)
    label = raw["label"]
    if label not in LABELS:
        raise DatasetSchemaError(
            f"unknown label {label!r}; expected one of {LABELS}",
            line_no=line_no,
            field_name="label",
        )


def iter_dataset(path: str | Path) -> Iterator[DatasetRecord]:
    """Stream records from a JSONL dataset, validating each line."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetSchemaError(f"invalid JSON: {exc.msg}", line_no=i) from exc
            yield DatasetRecord.from_dict(raw, line_no=i)


def write_dataset(path: str | Path, records: Iterable[DatasetRecord]) -> int:
    """Write records as JSONL. Returns the number of records written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n
