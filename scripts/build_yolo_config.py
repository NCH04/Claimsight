#!/usr/bin/env python3
"""Génère le data.yaml YOLO depuis la taxonomie.

L'ancien scripts/clean_yolo_parts.py dérivait les ids de l'ordre d'insertion
d'un dict Python, sans jamais écrire de data.yaml. Le mapping id <-> nom
n'était donc versionné nulle part: réordonner le dict décalait silencieusement
tous les labels. Ici, l'ordre de PART_CLASSES est l'unique autorité.

    python scripts/build_yolo_config.py --dataset_root dataset/parts_yolo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from claimsight.domain.taxonomy import PART_CLASSES


def build_yaml(dataset_root: Path) -> str:
    lines = [
        "# GÉNÉRÉ PAR scripts/build_yolo_config.py — NE PAS ÉDITER À LA MAIN.",
        "# Source de vérité: claimsight/domain/taxonomy.py::PART_CLASSES",
        f"path: {dataset_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(PART_CLASSES)}",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(PART_CLASSES)]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset_root", default="dataset/parts_yolo")
    ap.add_argument("--out", default="configs/parts_yolo.yaml")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_yaml(Path(args.dataset_root)), encoding="utf-8")
    print(f"{out} écrit — {len(PART_CLASSES)} classes.")


if __name__ == "__main__":
    main()
