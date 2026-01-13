#!/usr/bin/env python3
"""Count images per view type in the car damages dataset."""

import json
import sys
from collections import Counter
from pathlib import Path


def count_views(dataset_path: Path) -> Counter:
    with dataset_path.open() as f:
        data = json.load(f)

    counts = Counter()
    for entry in data:
        view = entry.get("view")
        if view:
            counts[view] += 1
    return counts


def main() -> None:
    dataset_path = Path(
        sys.argv[1] if len(sys.argv) > 1 else "data/final_dataset/car-damages-v1.json.json"
    )
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found at {dataset_path}")

    counts = count_views(dataset_path)
    print("Image count by view type:")
    for view, count in counts.most_common():
        print(f"{view}: {count}")


if __name__ == "__main__":
    main()
