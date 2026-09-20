"""Pipeline I/O."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ..domain.taxonomy import UNKNOWN  # noqa: F401  (ré-export pratique)
from ..training.dataset import load_rgb

#: Decompression-bomb guard: an 8000x8000 image decoded to RGB
#: fait ~190 Mo en RAM. On refuse au-delà de ce seuil plutôt que de faire
#: tomber le process.
MAX_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(images_dir: str | Path, recursive: bool = False) -> list[Path]:
    """List a folder's images, sorted so the order is deterministic."""
    root = Path(images_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dossier d'images introuvable: {images_dir}")

    allowed = {e.lower() for e in IMAGE_EXTS}
    it = root.rglob("*") if recursive else root.iterdir()
    files = sorted(p for p in it if p.is_file() and p.suffix.lower() in allowed)

    if not files:
        hint = "" if recursive else " (essayez recursive=True si les images sont en sous-dossiers)"
        raise ValueError(f"No image in {images_dir} with extensions {IMAGE_EXTS}{hint}")
    return files


def load_image(path: str | Path) -> Image.Image:
    """Charge une image en RGB avec correction de l'orientation EXIF.

    L'orientation EXIF est indispensable: une photo prise en portrait avec un
    smartphone est stockée en paysage + un flag de rotation. Sans correction,
    le classifieur voit le véhicule couché.
    """
    try:
        return load_rgb(path)
    except Exception as exc:
        raise ValueError(f"Could not load {path}: {exc}") from exc


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: str | Path) -> None:
    out_path = Path(path)
    ensure_parent(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
