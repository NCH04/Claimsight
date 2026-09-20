#!/usr/bin/env python3
"""Mesure le décalage train/serve entre images entières et crops de pièces.

Le classifieur de dégât est entraîné sur des images et servi sur des **crops**
produits par le détecteur. Si les deux domaines divergent, rien ne le signale:
la validation reste bonne, et c'est l'attribution par pièce qui ment.

Ce contrôle prend des véhicules supposés intacts et compare les deux réponses
sur les mêmes images. Un modèle sain répond `none` des deux côtés. Un modèle
entraîné uniquement sur images entières a répondu `none` 59 fois sur 60 en
image entière, et seulement 44 fois sur les 277 crops de ces mêmes images.

    python scripts/check_domain_skew.py --images_dir <véhicules intacts> \
        --damage_checkpoint models/damage.pt --parts_checkpoint models/parts.pt
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

from PIL import Image

from claimsight.ml.classifier import ImageClassifier
from claimsight.ml.detector import PartDetector

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def rate(labels: list[str]) -> float:
    return labels.count("none") / len(labels) if labels else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images_dir", required=True, help="Véhicules supposés intacts")
    ap.add_argument("--damage_checkpoint", default="models/damage.pt")
    ap.add_argument("--parts_checkpoint", default="models/parts.pt")
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_gap", type=float, default=0.25,
                    help="Écart toléré entre les deux taux de `none` (0 = désactivé)")
    args = ap.parse_args()

    clf = ImageClassifier(args.damage_checkpoint)
    det = PartDetector(args.parts_checkpoint)
    if not clf.available:
        sys.exit(f"Classifieur indisponible: {args.damage_checkpoint}")
    if not det.available:
        sys.exit(f"Détecteur indisponible: {args.parts_checkpoint}")

    paths = sorted(p for p in Path(args.images_dir).rglob("*")
                   if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        sys.exit(f"Aucune image dans {args.images_dir}")
    if args.sample and len(paths) > args.sample:
        paths = random.Random(args.seed).sample(paths, args.sample)
    images = [Image.open(p).convert("RGB") for p in paths]

    whole = [p.label for p in clf.predict_batch(images, args.batch_size)]
    crops = [d.crop(im) for im, dets in zip(images, det.detect(images, 8), strict=True) for d in dets]
    crop_labels = [p.label for p in clf.predict_batch(crops, args.batch_size)] if crops else []

    print(f"Images          : {len(images)}   ({args.images_dir})")
    print(f"Crops de pièces : {len(crop_labels)}\n")
    print(f"{'domaine':16} {'`none`':>12}   distribution")
    print(f"{'image entière':16} {whole.count('none'):>5}/{len(whole):<6} "
          f"{dict(collections.Counter(whole).most_common())}")
    print(f"{'crop de pièce':16} {crop_labels.count('none'):>5}/{len(crop_labels):<6} "
          f"{dict(collections.Counter(crop_labels).most_common())}")

    gap = abs(rate(whole) - rate(crop_labels))
    print(f"\nÉcart de taux `none` : {gap:.3f}")
    if args.max_gap and gap > args.max_gap:
        sys.exit(f"ÉCHEC — écart > {args.max_gap:.2f}: le modèle ne voit pas "
                 f"les crops comme il voit les images.")
    print("OK — les deux domaines concordent.")


if __name__ == "__main__":
    main()
