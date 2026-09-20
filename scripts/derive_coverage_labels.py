#!/usr/bin/env python3
"""Déduit des labels de couverture photo depuis un dataset de pièces YOLO.

Les pièces visibles sur une image trahissent l'angle de prise de vue : un
cliché qui annote un pare-chocs avant et des optiques avant documente la face
avant. Cette supervision faible fournit des milliers d'exemples gratuits, sans
annoter une seule image à la main.

C'est bien de la supervision FAIBLE : l'absence d'une pièce ne prouve pas
l'absence de la face (occlusion, cadrage, annotation incomplète). Le jeu obtenu
sert de socle — il ne remplace pas quelques centaines de vraies photos
annotées, surtout pour les flancs, rares dans les datasets publics.

    python scripts/derive_coverage_labels.py \
        --dataset /chemin/carparts-seg --out dataset/coverage_derived.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
from pathlib import Path

import yaml

from claimsight.domain.taxonomy import COVERAGE_FACES

#: Pièces attestant qu'une face est photographiée.
#: Les optiques gauche/droite et les rétroviseurs se voient DE FACE : ils
#: indiquent l'avant ou l'arrière, pas le flanc. Seules les PORTES attestent
#: réellement qu'un flanc est dans le cadre.
FACE_INDICATORS: dict[str, set[str]] = {
    "front": {"front_bumper", "front_glass", "hood", "front_light",
              "front_left_light", "front_right_light"},
    "rear": {"back_bumper", "back_glass", "tailgate", "trunk", "back_light",
             "back_left_light", "back_right_light"},
    "left": {"front_left_door", "back_left_door"},
    "right": {"front_right_door", "back_right_door"},
}


def faces_from_parts(parts: set[str]) -> list[str]:
    return [f for f in COVERAGE_FACES if parts & FACE_INDICATORS[f]]


def photo_id(filename: str) -> str:
    """Identifiant de la PHOTO d'origine, augmentations mises de côté.

    Un dataset Roboflow contient ~6 variantes augmentées par photo, nommées
    `<photo>_jpg.rf.<hash>.jpg`. Sans cet identifiant, les variantes d'une même
    photo se répartissent entre entraînement et validation: le modèle revoit en
    validation ce qu'il a appris, et les métriques sont surévaluées. C'est la
    colonne à passer en `--group_column`.
    """
    return re.split(r"_(jpg|png|jpeg)", Path(filename).name, maxsplit=1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Racine du dataset YOLO source")
    ap.add_argument("--yaml", default=None, help="data.yaml du source (défaut: <dataset>/*.yaml)")
    ap.add_argument("--out", default="dataset/coverage_derived.csv")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    root = Path(args.dataset)
    cfg = Path(args.yaml) if args.yaml else next(root.glob("*.yaml"))
    names = yaml.safe_load(cfg.read_text(encoding="utf-8"))["names"]

    rows, stats = [], collections.Counter()
    for split in args.splits:
        for label_file in sorted((root / "labels" / split).glob("*.txt")):
            ids = {int(line.split()[0])
                   for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()}
            if not ids:
                stats["vide"] += 1
                continue
            faces = faces_from_parts({names[i] for i in ids if i in names})
            if not faces:
                stats["aucune face"] += 1
                continue
            image = next((p for p in (root / "images" / split).glob(f"{label_file.stem}.*")), None)
            if image is None:
                stats["image absente"] += 1
                continue
            rows.append({"image": str(image.resolve()),
                         **{f: int(f in faces) for f in COVERAGE_FACES},
                         "photo_id": photo_id(image.name),
                         "source": "derived"})
            for f in faces:
                stats[f] += 1
            stats["retenues"] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", *COVERAGE_FACES, "photo_id", "source"])
        w.writeheader()
        w.writerows(rows)

    photos = {r["photo_id"] for r in rows}
    print(f"{out} — {len(rows)} images / {len(photos)} photos distinctes "
          f"({len(rows) / max(len(photos), 1):.1f} augmentations par photo)")
    for face in COVERAGE_FACES:
        pct = 100 * stats[face] / max(len(rows), 1)
        print(f"  {face:6s} {stats[face]:5d}  ({pct:.0f} %)")
    for key in ("vide", "aucune face", "image absente"):
        if stats[key]:
            print(f"  ignorées ({key}): {stats[key]}")


if __name__ == "__main__":
    main()
