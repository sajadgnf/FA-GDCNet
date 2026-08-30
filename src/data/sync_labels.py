"""Copy dataset jsonl labels onto the feature-cache ``y`` vector.

Does not recompute ``X``. Run after a human relabel pass, then train from cache.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data.relabel import DEFAULT_DATASET, DEFAULT_FEATURES, sync_feature_labels

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync feature-cache labels from jsonl.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--features-cache", type=Path, default=DEFAULT_FEATURES)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    changed = sync_feature_labels(args.dataset, args.features_cache)
    log.info("updated %d labels in %s", changed, args.features_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
