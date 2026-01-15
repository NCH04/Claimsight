from dataclasses import dataclass
from typing import List, Tuple

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Extensions d'image supportées (case-insensitive)
IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Taxonomie (source de vérité V1)
VIEW_CLASSES: List[str] = [
    "front",
    "rear",
    "left",
    "right",
    "front-left",
    "front-right",
    "rear-left",
    "rear-right",
    "closeup",
    "out_of_scope",
]

DEFAULT_DAMAGE_CLASSES: List[str] = [
    "none",
    "scratch",
    "dent",
    "crack",
    "broken_glass",
    "deformation_impact",
    "missing_part",
    "unknown",
]

DEFAULT_SEVERITY_CLASSES: List[str] = ["none", "minor", "moderate", "severe", "unknown"]

# Poids simples pour la confiance globale (view + damage)
CONFIDENCE_WEIGHTS = {
    "view": 0.5,
    "damage": 0.5,
}


@dataclass
class CheckpointMeta:
    arch: str
    classes: List[str]
    img_size: int
    normalize: str
