import json
import os
from pathlib import Path
from typing import List, Optional

from PIL import Image

from .config import IMAGE_EXTS


def list_images(images_dir: str) -> List[Path]:
    root = Path(images_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    allowed = {e.lower() for e in IMAGE_EXTS}
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in allowed]
    files.sort()
    if not files:
        raise ValueError(f"No images found in {images_dir} with extensions {IMAGE_EXTS}")
    return files


def load_image(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {e}") from e


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: str) -> None:
    out_path = Path(path)
    ensure_parent(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
