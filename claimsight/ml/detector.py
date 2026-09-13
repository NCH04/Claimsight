"""Détecteur de pièces (YOLO), enveloppé pour le pipeline.

`ultralytics` est une dépendance optionnelle (`pip install -e ".[yolo]"`).
Comme `ImageClassifier`, l'objet reste utilisable sans poids ni bibliothèque:
il rapporte `available = False` et ne renvoie aucune détection, ce qui permet
au pipeline de tourner en dégradé plutôt que de refuser de démarrer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """Une boîte détectée sur une image."""

    label: str
    confidence: float
    #: [x1, y1, x2, y2] en pixels de l'image d'origine.
    bbox: tuple[float, float, float, float]

    def as_dict(self, ndigits: int = 4) -> dict:
        return {
            "label": self.label,
            "score": round(self.confidence, ndigits),
            "bbox": [round(v, 1) for v in self.bbox],
        }

    def crop(self, image: Image.Image, margin: float = 0.08) -> Image.Image:
        """Découpe la pièce, avec une marge pour garder un peu de contexte.

        Le contexte compte: un enfoncement se lit en partie à la déformation
        des arêtes voisines, qu'un recadrage au ras de la boîte supprimerait.
        """
        x1, y1, x2, y2 = self.bbox
        dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
        box = (
            max(0, int(x1 - dx)), max(0, int(y1 - dy)),
            min(image.width, int(x2 + dx)), min(image.height, int(y2 + dy)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            return image
        return image.crop(box)


class PartDetector:
    """Détecteur de pièces servi depuis un checkpoint ultralytics."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        min_confidence: float = 0.35,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.min_confidence = min_confidence
        self.device = device
        self.model = None
        self.available = False

        if checkpoint_path is None or not Path(checkpoint_path).exists():
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            LOGGER.warning(
                "ultralytics n'est pas installé -> détection de pièces désactivée. "
                'Installez-le avec: pip install -e ".[yolo]"'
            )
            return

        self.model = YOLO(str(checkpoint_path))
        self.names = dict(self.model.names)
        self.available = True

    def __repr__(self) -> str:  # pragma: no cover - confort de debug
        if not self.available:
            return "<PartDetector unavailable>"
        return f"<PartDetector classes={len(self.names)} conf>={self.min_confidence}>"

    def detect(
        self, images: list[Image.Image], batch_size: int = 8
    ) -> list[list[Detection]]:
        """Détections par image, dans l'ordre d'entrée."""
        if not self.available or not images:
            return [[] for _ in images]

        out: list[list[Detection]] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            results = self.model.predict(
                chunk, conf=self.min_confidence, device=self.device, verbose=False
            )
            for res in results:
                dets = []
                for box in res.boxes:
                    xyxy = [float(v) for v in box.xyxy[0]]
                    dets.append(Detection(
                        label=self.names[int(box.cls)],
                        confidence=float(box.conf),
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    ))
                out.append(dets)
        return out
