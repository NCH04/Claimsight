import os
import random
import shutil
from pathlib import Path

ROOT = Path("dataset/parts_yolo")
IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"

TRAIN_IMAGES = IMAGES_DIR / "train"
VAL_IMAGES = IMAGES_DIR / "val"
TRAIN_LABELS = LABELS_DIR / "train"
VAL_LABELS = LABELS_DIR / "val"

# Dossiers temporaires pour regrouper toutes les paires avant de re-split
STAGING_IMAGES = ROOT / "_staging_images"
STAGING_LABELS = ROOT / "_staging_labels"

VAL_RATIO = 0.2
EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp", ".BMP", ".webp", ".jfif"}


def ensure_empty(dir_path: Path):
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


def collect_pairs():
    """Rassemble images/labels depuis images/, images/train, images/val (dedup)."""
    src_img_dirs = [IMAGES_DIR, IMAGES_DIR / "train", IMAGES_DIR / "val"]
    src_lbl_dirs = [LABELS_DIR, LABELS_DIR / "train", LABELS_DIR / "val"]

    def find_label(stem):
        for d in src_lbl_dirs:
            cand = d / f"{stem}.txt"
            if cand.exists():
                return cand
        return None

    ensure_empty(STAGING_IMAGES)
    ensure_empty(STAGING_LABELS)

    pairs = []
    seen = set()
    dup_stems = set()
    missing_lbl = []

    for img_dir in src_img_dirs:
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if not img_path.is_file():
                continue
            if img_path.suffix not in EXTS:
                continue
            stem = img_path.stem
            if stem in seen:
                dup_stems.add(stem)
                continue
            lbl = find_label(stem)
            if lbl is None:
                missing_lbl.append(img_path.name)
                continue

            seen.add(stem)
            dst_img = STAGING_IMAGES / img_path.name
            dst_lbl = STAGING_LABELS / f"{stem}.txt"
            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl, dst_lbl)
            pairs.append((dst_img, dst_lbl))

    return pairs, dup_stems, missing_lbl


def main():
    pairs, dup_stems, missing_lbl = collect_pairs()
    print(f"Total paires valides: {len(pairs)}")
    if dup_stems:
        print(f"Doublons ignorés: {len(dup_stems)}")
    if missing_lbl:
        print(f"Images sans label ignorées: {len(missing_lbl)}")

    # Réinitialise les dossiers train/val
    for d in [TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS]:
        ensure_empty(d)

    random.seed(42)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_RATIO)) if pairs else 0
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    def move_pair(pair, dst_img_dir: Path, dst_lbl_dir: Path):
        img_path, lbl_path = pair
        shutil.move(str(img_path), dst_img_dir / img_path.name)
        shutil.move(str(lbl_path), dst_lbl_dir / lbl_path.name)

    for p in train_pairs:
        move_pair(p, TRAIN_IMAGES, TRAIN_LABELS)
    for p in val_pairs:
        move_pair(p, VAL_IMAGES, VAL_LABELS)

    # Nettoyage staging
    if STAGING_IMAGES.exists():
        shutil.rmtree(STAGING_IMAGES)
    if STAGING_LABELS.exists():
        shutil.rmtree(STAGING_LABELS)

    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")


if __name__ == "__main__":
    main()
