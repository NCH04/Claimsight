#!/usr/bin/env python3
"""Déduit des labels image-level de dégât et de gravité depuis un dataset YOLO.

Les classes du dataset source encodent souvent les deux informations à la fois:
`severe-deformation` donne le type ET le niveau. La correspondance vit dans un
fichier de configuration, pas dans le code.

Une image porte souvent plusieurs boîtes. On retient la PIRE constatation, par
gravité ordinale — c'est la règle du §3.4 des consignes d'annotation, et celle
que le pipeline applique déjà à l'échelle du dossier.

    python scripts/derive_damage_labels.py \
        --dataset ../datasets/car-damage-severity \
        --none_from ../datasets/carparts-seg/images/train \
        --out dataset/damage_derived.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
from pathlib import Path

import yaml

from claimsight.domain.taxonomy import severity_rank

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def load_mapping(path: Path) -> dict[str, dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))["mapping"]


def photo_id(filename: str) -> str:
    """Identifiant de la photo d'origine, augmentations Roboflow mises de côté.

    Sans lui, les variantes d'une même photo se répartissent entre
    entraînement et validation et gonflent les métriques.
    """
    return re.split(r"[_.]rf[._]", Path(filename).name)[0]


def worst(findings: list[tuple[str, str]]) -> tuple[str, str]:
    """Pire constatation d'une image: gravité ordinale d'abord."""
    return max(findings, key=lambda ds: severity_rank(ds[1]))


def collect(root: Path, mapping: dict, splits: list[str]) -> tuple[list[dict], collections.Counter]:
    names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
    rows, stats = [], collections.Counter()

    for split in splits:
        label_dir = root / split / "labels"
        image_dir = root / split / "images"
        if not label_dir.is_dir():
            continue
        for label_file in sorted(label_dir.glob("*.txt")):
            findings = []
            for line in label_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    source_class = names[int(line.split()[0])]
                except (ValueError, IndexError):
                    stats["malformé"] += 1
                    continue
                target = mapping.get(source_class)
                if target is None:
                    stats[f"ignoré:{source_class}"] += 1
                    continue
                findings.append((target["damage"], target["severity"]))
            if not findings:
                stats["sans dégât exploitable"] += 1
                continue
            image = next((p for p in image_dir.glob(f"{label_file.stem}.*")
                          if p.suffix.lower() in IMAGE_EXTS), None)
            if image is None:
                stats["image absente"] += 1
                continue
            damage, severity = worst(findings)
            rows.append({"image": str(image.resolve()), "damage": damage,
                         "severity": severity, "n_findings": len(findings),
                         "photo_id": photo_id(image.name), "source": "derived"})
            stats[f"{damage}/{severity}"] += 1
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Racine du dataset de dégâts (YOLO)")
    ap.add_argument("--mapping", default="configs/source_damage_ids.json")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--none_from", default=None,
                    help="Dossier de véhicules INTACTS, pour la classe `none` "
                         "que le dataset de dégâts ne contient pas")
    ap.add_argument("--none_ratio", type=float, default=0.30,
                    help="Proportion d'exemples `none` dans le jeu final")
    ap.add_argument("--out", default="dataset/damage_derived.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mapping = load_mapping(Path(args.mapping))
    rows, stats = collect(Path(args.dataset), mapping, args.splits)
    n_damaged = len(rows)

    if args.none_from:
        # Le dataset de dégâts ne contient QUE des véhicules abîmés: sans
        # contre-exemples, le classifieur ne saurait jamais répondre « rien ».
        pool = [p for p in Path(args.none_from).iterdir() if p.suffix.lower() in IMAGE_EXTS]
        n_none = min(len(pool), int(n_damaged * args.none_ratio / (1 - args.none_ratio)))
        for p in random.Random(args.seed).sample(pool, n_none):
            rows.append({"image": str(p.resolve()), "damage": "none", "severity": "none",
                         "n_findings": 0, "photo_id": photo_id(p.name),
                         "source": "assumed_intact"})
        stats["none/none"] = n_none

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "damage", "severity",
                                           "n_findings", "photo_id", "source"])
        w.writeheader()
        w.writerows(rows)

    photos = {r["photo_id"] for r in rows}
    print(f"{out} — {len(rows)} images / {len(photos)} photos ({n_damaged} endommagées)")
    for key, count in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {key:28s} {count}")


if __name__ == "__main__":
    main()
