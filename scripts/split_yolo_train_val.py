#!/usr/bin/env python3
"""Split train/val d'un dataset YOLO.

Différences avec la version précédente:
  - non destructif: écrit dans un dossier de sortie distinct au lieu de
    `rmtree` les dossiers train/val existants;
  - groupable (`--group_regex`) pour éviter que deux photos du même véhicule
    se retrouvent de part et d'autre du split;
  - extensions insensibles à la casse;
  - `--dry_run`.

    python scripts/split_yolo_train_val.py --root dataset/parts_yolo --dry_run
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


def collect_pairs(images_root: Path, labels_root: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Apparie chaque image à son .txt, en cherchant aussi dans train/ et val/."""
    img_dirs = [images_root, images_root / "train", images_root / "val"]
    lbl_dirs = [labels_root, labels_root / "train", labels_root / "val"]

    def find_label(stem: str) -> Path | None:
        for d in lbl_dirs:
            candidate = d / f"{stem}.txt"
            if candidate.exists():
                return candidate
        return None

    pairs: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    orphans: list[str] = []

    for img_dir in img_dirs:
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in EXTS:
                continue
            if img.stem in seen:
                continue
            label = find_label(img.stem)
            if label is None:
                orphans.append(img.name)
                continue
            seen.add(img.stem)
            pairs.append((img, label))
    return pairs, orphans


def group_of(path: Path, pattern: re.Pattern | None) -> str:
    """Clé de groupe d'une image; le stem lui-même si aucun motif fourni."""
    if pattern is None:
        return path.stem
    m = pattern.search(path.stem)
    return m.group(1) if m and m.groups() else path.stem


def split_by_group(
    pairs: list[tuple[Path, Path]], val_ratio: float, seed: int, pattern: re.Pattern | None
):
    """Assigne des GROUPES entiers à val, jamais des images isolées."""
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for img, lbl in pairs:
        groups[group_of(img, pattern)].append((img, lbl))

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    target = len(pairs) * val_ratio
    val_pairs, train_pairs, count = [], [], 0
    for key in keys:
        if count < target:
            val_pairs.extend(groups[key])
            count += len(groups[key])
        else:
            train_pairs.extend(groups[key])
    return train_pairs, val_pairs, len(groups)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset/parts_yolo", help="Racine du dataset source")
    ap.add_argument("--out", default=None, help="Racine de sortie (défaut: <root>_split)")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group_regex", default=None,
                    help=r"Regex à 1 groupe capturant l'id véhicule dans le nom de fichier, ex: '^(VEH\d+)'")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out) if args.out else root.with_name(root.name + "_split")
    if out_root.resolve() == root.resolve():
        raise SystemExit("--out doit différer de --root (pas de réécriture en place).")

    pairs, orphans = collect_pairs(root / "images", root / "labels")
    if not pairs:
        raise SystemExit(f"Aucune paire image/label trouvée sous {root}")

    pattern = re.compile(args.group_regex) if args.group_regex else None
    if pattern is None:
        print("[AVERTISSEMENT] Aucun --group_regex: split par image. Si plusieurs photos "
              "partagent un véhicule, la validation sera optimiste.")

    train_pairs, val_pairs, n_groups = split_by_group(pairs, args.val_ratio, args.seed, pattern)

    print(f"Paires: {len(pairs)} | groupes: {n_groups} | orphelines ignorées: {len(orphans)}")
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    if args.dry_run:
        print("\n[DRY RUN] Rien n'a été écrit.")
        return

    for split, items in (("train", train_pairs), ("val", val_pairs)):
        img_dst = out_root / "images" / split
        lbl_dst = out_root / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, img_dst / img.name)
            shutil.copy2(lbl, lbl_dst / lbl.name)

    print(f"\nÉcrit dans {out_root} (source intacte).")
    print(f"Pensez à: python scripts/build_yolo_config.py --dataset_root {out_root}")


if __name__ == "__main__":
    main()
