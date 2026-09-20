#!/usr/bin/env python3
"""Construit un jeu d'exemples au niveau *crop de pièce*, pas image entière.

Pourquoi. Le pipeline sert le classifieur de dégât sur des **crops** produits
par le détecteur de pièces, alors qu'il était entraîné sur des **images
entières**. Le décalage est mesurable et sévère: sur 60 images de validation,
le modèle image-entière répond `none` 59 fois, mais seulement 44 fois sur les
277 crops issus de ces mêmes images. Il voit des dégâts partout.

La correction consiste à entraîner sur les deux domaines. Ce script produit la
moitié manquante:

  * positifs — chaque polygone de dégât du dataset source, recadré sur sa boîte
    englobante, hérite du (type, gravité) de sa classe. Un crop = une
    constatation, donc aucune règle d'agrégation ici.
  * négatifs — chaque pièce annotée d'un véhicule intact, recadrée pareil,
    devient un exemple `none` dans le domaine du service.

La marge de recadrage reprend `Detection.crop` (8 %): un enfoncement se lit en
partie à la déformation des arêtes voisines.

    python scripts/derive_damage_crops.py \
        --dataset ../datasets/car-damage-severity \
        --intact_from ../datasets/claimsight-parts-v2 \
        --out dataset/damage_crops.csv --crops_dir dataset/crops
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path

from PIL import Image

from claimsight.ml.detector import Detection

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
#: Même marge que `Detection.crop`, pour que l'entraînement voie ce que le
#: service produit.
CROP_MARGIN = 0.08
#: En deçà, le crop ne porte plus assez de pixels pour être classé.
MIN_CROP_PX = 24


def photo_id(filename: str) -> str:
    """Identifiant de la photo d'origine, augmentations Roboflow mises de côté."""
    import re

    return re.split(r"[_.]rf[._]", Path(filename).name)[0]


def polygon_bbox(parts: list[str], width: int, height: int) -> tuple[float, float, float, float] | None:
    """Boîte englobante d'un polygone YOLO normalisé, en pixels."""
    coords = [float(v) for v in parts]
    xs, ys = coords[0::2], coords[1::2]
    if not xs or not ys:
        return None
    return (min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height)


def iter_labelled(root: Path, splits: list[str]):
    """(chemin image, chemin label) pour chaque image annotée."""
    for split in splits:
        images, labels = root / split / "images", root / split / "labels"
        if not images.is_dir():
            images, labels = root / "images" / split, root / "labels" / split
        if not images.is_dir():
            continue
        for img in sorted(images.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lab = labels / f"{img.stem}.txt"
            if lab.exists():
                yield img, lab


def cut(img_path: Path, lab_path: Path, out_dir: Path, label_of, source: str,
        stats: collections.Counter) -> list[dict]:
    """Découpe une image en un crop par annotation retenue."""
    try:
        image = Image.open(img_path).convert("RGB")
    except OSError:
        stats["image_illisible"] += 1
        return []

    rows = []
    for i, line in enumerate(lab_path.read_text().split("\n")):
        fields = line.split()
        if len(fields) < 7:  # class + au moins 3 points
            continue
        labels = label_of(int(fields[0]))
        if labels is None:
            stats["classe_ignorée"] += 1
            continue
        bbox = polygon_bbox(fields[1:], image.width, image.height)
        if bbox is None:
            continue
        crop = Detection(label="", confidence=1.0, bbox=bbox).crop(image, CROP_MARGIN)
        if crop.width < MIN_CROP_PX or crop.height < MIN_CROP_PX:
            stats["crop_trop_petit"] += 1
            continue
        dst = out_dir / f"{img_path.stem}__{i}.jpg"
        crop.save(dst, quality=92)
        damage, severity = labels
        rows.append({
            "image": str(dst.resolve()),
            "damage": damage,
            "severity": severity,
            "n_findings": 1,
            "photo_id": photo_id(img_path.name),
            "source": source,
        })
        stats[damage] += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Dataset de dégâts (YOLO, polygones)")
    ap.add_argument("--mapping", default="configs/source_damage_ids.json")
    ap.add_argument("--intact_from", default=None,
                    help="Dataset de pièces: ses véhicules fournissent les crops `none`")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--intact_splits", nargs="+", default=["train"])
    ap.add_argument("--max_intact", type=int, default=2500,
                    help="Plafond de crops `none` (0 = pas de plafond)")
    ap.add_argument("--out", default="dataset/damage_crops.csv")
    ap.add_argument("--crops_dir", default="dataset/crops")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))["mapping"]
    data_yaml = Path(args.dataset) / "data.yaml"
    import yaml
    names = yaml.safe_load(data_yaml.read_text())["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else names

    def damage_label(cid: int):
        entry = mapping.get(names.get(cid, ""))
        return (entry["damage"], entry["severity"]) if entry else None

    out_dir = Path(args.crops_dir)
    (out_dir / "damage").mkdir(parents=True, exist_ok=True)
    (out_dir / "intact").mkdir(parents=True, exist_ok=True)

    stats: collections.Counter = collections.Counter()
    rows: list[dict] = []

    for img, lab in iter_labelled(Path(args.dataset), args.splits):
        rows += cut(img, lab, out_dir / "damage", damage_label, "crop_damage", stats)
    print(f"Crops de dégâts : {len(rows)}")

    if args.intact_from:
        intact: list[dict] = []
        for img, lab in iter_labelled(Path(args.intact_from), args.intact_splits):
            intact += cut(img, lab, out_dir / "intact",
                          lambda _cid: ("none", "none"), "crop_intact", stats)
        if args.max_intact and len(intact) > args.max_intact:
            # Échantillonner par photo, pas par crop: garder des photos entières
            # évite qu'un même véhicule soit à moitié dedans, à moitié dehors.
            random.Random(args.seed).shuffle(intact)
            intact = intact[: args.max_intact]
        rows += intact
        print(f"Crops intacts   : {len(intact)}")

    if not rows:
        raise SystemExit("Aucun crop produit — vérifiez --dataset et --splits.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{out} — {len(rows)} lignes, {len({r['photo_id'] for r in rows})} photos")
    for k, v in sorted(stats.items()):
        print(f"  {k:22} {v}")


if __name__ == "__main__":
    main()
