"""Data acquisition, preprocessing, labeling, and schema validation."""

from .preprocess import is_persian_enough, normalize_persian, preprocess_caption
from .schema import (
    DatasetRecord,
    DatasetSchemaError,
    LABELS,
    iter_dataset,
    validate_record,
)

__all__ = [
    "DatasetRecord",
    "DatasetSchemaError",
    "LABELS",
    "is_persian_enough",
    "iter_dataset",
    "normalize_persian",
    "preprocess_caption",
    "validate_record",
]
