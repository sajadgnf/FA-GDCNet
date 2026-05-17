"""LanceDB-backed vector store for per-sample embeddings.

Stores `(post_id, label, image_emb, text_emb, T_hat_emb, discrepancy_vec)` so
the evaluation and ablation scripts can pull batches of features without
re-running the heavy backbones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

DEFAULT_DB_PATH = Path("artifacts") / "lancedb"
DEFAULT_TABLE = "samples"


def _connect(db_path: Path):
    import lancedb  # lazy

    db_path.parent.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(db_path))


def _to_row(rec: dict) -> dict[str, Any]:
    required = (
        "post_id",
        "label",
        "image_emb",
        "text_emb",
        "T_hat_emb",
        "discrepancy_vec",
    )
    for k in required:
        if k not in rec:
            raise KeyError(f"vector store row is missing required field {k!r}")
    return {k: rec[k] for k in required}


def upsert(records: Iterable[dict], *, db_path: Path = DEFAULT_DB_PATH, table: str = DEFAULT_TABLE) -> int:
    """Add or replace rows keyed by `post_id`. Returns the number written."""
    db = _connect(db_path)
    rows = [_to_row(r) for r in records]
    if not rows:
        return 0
    if table in db.table_names():
        t = db.open_table(table)
        # Delete pre-existing rows for these post_ids, then insert.
        ids = ", ".join(f"'{r['post_id']}'" for r in rows)
        t.delete(f"post_id IN ({ids})")
        t.add(rows)
    else:
        db.create_table(table, data=rows)
    return len(rows)


def fetch_all(*, db_path: Path = DEFAULT_DB_PATH, table: str = DEFAULT_TABLE) -> list[dict]:
    """Return every row as a list of dicts."""
    db = _connect(db_path)
    if table not in db.table_names():
        return []
    t = db.open_table(table)
    return t.to_pandas().to_dict(orient="records")
