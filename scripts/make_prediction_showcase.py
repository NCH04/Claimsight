#!/usr/bin/env python3
"""Compose une planche de prédictions du détecteur de pièces, pour le README.

Les planches produites par ultralytics pendant l'entraînement conviennent au
débogage, pas à une page d'accueil: elles incarnent le nom de fichier en haut
de chaque vignette, et les noms Roboflow se chevauchent. On redessine donc
proprement, à partir des mêmes poids et des mêmes images de validation.

    python scripts/make_prediction_showcase.py \
        --images_dir ../datasets/claimsight-parts-v2/images/val \
        --out docs/assets/parts-predictions.jpg
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def looks_rotated(image: Image.Image, patch: int = 40, flat: float = 6.0) -> bool:
    """Une rotation Roboflow laisse des coins uniformes, noirs ou blancs.

    Une photo réelle a du contenu dans ses coins; un aplat parfait sur trois
    coins ou plus trahit une augmentation, qui fait mauvais effet en vitrine.
    """
    from PIL import ImageStat

    w, h = image.size
    boxes = ((0, 0, patch, patch), (w - patch, 0, w, patch),
             (0, h - patch, patch, h), (w - patch, h - patch, w, h))
    uniform = 0
    for box in boxes:
        stat = ImageStat.Stat(image.crop(box).convert("L"))
        if stat.stddev[0] < flat and (stat.mean[0] < 40 or stat.mean[0] > 215):
            uniform += 1
    return uniform >= 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--checkpoint", default="models/parts.pt")
    ap.add_argument("--out", default="docs/assets/parts-predictions.jpg")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cell", type=int, default=640, help="Côté d'une vignette, en pixels")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--min_parts", type=int, default=3,
                    help="N'afficher que des images où le modèle trouve au moins N pièces")
    ap.add_argument("--allow_rotated", action="store_true",
                    help="Garder les copies pivotées de Roboflow (coins remplis)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.checkpoint)
    paths = sorted(p for p in Path(args.images_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    random.Random(args.seed).shuffle(paths)

    wanted = args.cols * args.rows
    tiles: list[Image.Image] = []
    for path in paths:
        if len(tiles) == wanted:
            break
        if not args.allow_rotated and looks_rotated(Image.open(path).convert("RGB")):
            continue
        res = model.predict(str(path), conf=args.conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) < args.min_parts:
            continue
        # `plot()` rend boîtes et masques sans incruster le nom de fichier.
        tile = Image.fromarray(res.plot(labels=True, boxes=True, line_width=3)[:, :, ::-1])
        side = min(tile.size)
        left, top = (tile.width - side) // 2, (tile.height - side) // 2
        tiles.append(tile.crop((left, top, left + side, top + side))
                     .resize((args.cell, args.cell), Image.LANCZOS))

    if len(tiles) < wanted:
        raise SystemExit(f"Seulement {len(tiles)} images retenues sur {wanted} "
                         f"(baissez --min_parts ou --conf).")

    gap = 8
    sheet = Image.new(
        "RGB",
        (args.cols * args.cell + (args.cols - 1) * gap,
         args.rows * args.cell + (args.rows - 1) * gap),
        (255, 255, 255),
    )
    for i, tile in enumerate(tiles):
        x, y = i % args.cols, i // args.cols
        sheet.paste(tile, (x * (args.cell + gap), y * (args.cell + gap)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88, optimize=True)
    print(f"{out} — {sheet.width}x{sheet.height}, {len(tiles)} images, "
          f"{out.stat().st_size / 1024:.0f} Ko")


if __name__ == "__main__":
    main()
