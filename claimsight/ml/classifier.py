"""Classifieur d'images générique.

Remplace ViewModel / DamageModel / SeverityModel, qui étaient trois classes
quasi identiques (~80 lignes chacune) ne différant que par leur liste de
classes par défaut et leur message d'erreur.

Supporte le mode "indisponible": si aucun checkpoint n'est fourni, l'objet
reste utilisable et renvoie `unknown` avec une confiance nulle, ce qui permet
au pipeline de tourner tant que les modèles damage/severity ne sont pas
entraînés.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image

from ..domain.taxonomy import UNKNOWN
from ..domain.types import Prediction
from .backbones import build_backbone
from .checkpoint import CheckpointMeta, load_checkpoint
from .transforms import build_eval_transform

UNAVAILABLE = Prediction(UNKNOWN, 0.0)


def resolve_device(device: str | None = None) -> torch.device:
    """Résout le device au moment de l'appel (et non à l'import du module)."""
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageClassifier:
    """Classifieur mono-label servi depuis un checkpoint auto-descriptif."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        min_confidence: float = 0.0,
        task: str = "classifier",
    ) -> None:
        self.task = task
        self.min_confidence = min_confidence
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.device = resolve_device(device)
        self.model = None
        self.meta: CheckpointMeta | None = None
        self.transform = None

        if checkpoint_path is None or not Path(checkpoint_path).exists():
            self.available = False
            return

        state, meta = load_checkpoint(checkpoint_path)
        model = build_backbone(meta.arch, num_classes=len(meta.classes), pretrained=False)
        model.load_state_dict(state)
        model.to(self.device).eval()

        self.model = model
        self.meta = meta
        self.transform = build_eval_transform(
            img_size=meta.img_size,
            resize_mode=meta.resize_mode,
            normalize=meta.normalize,
        )
        self.available = True

    # -- introspection ------------------------------------------------------

    @property
    def classes(self) -> list[str]:
        return list(self.meta.classes) if self.meta else []

    def __repr__(self) -> str:  # pragma: no cover - confort de debug
        if not self.available:
            return f"<ImageClassifier task={self.task!r} unavailable>"
        return (
            f"<ImageClassifier task={self.task!r} arch={self.meta.arch} "
            f"classes={len(self.meta.classes)} img_size={self.meta.img_size} "
            f"resize={self.meta.resize_mode} device={self.device}>"
        )

    # -- inférence ----------------------------------------------------------

    def _postprocess(self, probs: torch.Tensor) -> Prediction:
        conf, idx = torch.max(probs, dim=0)
        confidence = float(conf)
        if confidence < self.min_confidence:
            return Prediction(UNKNOWN, confidence)
        return Prediction(self.meta.classes[int(idx)], confidence)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        if not self.available:
            return UNAVAILABLE
        x = self.transform(image).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0]
        return self._postprocess(probs)

    @torch.inference_mode()
    def predict_batch(
        self, images: Sequence[Image.Image], batch_size: int = 16
    ) -> list[Prediction]:
        """Inférence par lots: une passe pour N images au lieu de N passes."""
        if not self.available:
            return [UNAVAILABLE] * len(images)
        out: list[Prediction] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            batch = torch.stack([self.transform(im) for im in chunk]).to(self.device)
            probs = torch.softmax(self.model(batch) / self.temperature, dim=1)
            out.extend(self._postprocess(row) for row in probs)
        return out

    # -- multi-label ---------------------------------------------------------

    @property
    def multilabel(self) -> bool:
        return bool(self.meta and self.meta.multilabel)

    @property
    def temperature(self) -> float:
        return self.meta.temperature if self.meta else 1.0

    def threshold_for(self, cls: str, default: float) -> float:
        """Seuil calibré de cette classe, ou le seuil uniforme à défaut."""
        return self.meta.thresholds.get(cls, default) if self.meta else default

    @torch.inference_mode()
    def predict_labels(
        self, images: Sequence[Image.Image], threshold: float = 0.5, batch_size: int = 16
    ) -> list[list[Prediction]]:
        """Classes présentes sur chaque image, pour un modèle multi-label.

        Chaque classe est indépendante: on applique une sigmoïde par sortie et
        on retient celles au-dessus du seuil. Une photo peut donc documenter
        deux faces à la fois, ce qu'un argmax rendrait impossible.
        """
        if not self.available:
            return [[] for _ in images]
        if not self.multilabel:
            raise ValueError(
                f"predict_labels attend un checkpoint multi-label; "
                f"{self.task!r} est mono-label. Utilisez predict()."
            )
        out: list[list[Prediction]] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            batch = torch.stack([self.transform(im) for im in chunk]).to(self.device)
            # La température calibre les probabilités; les seuils par classe
            # tiennent compte du fait qu'une face rare gagne à être déclarée
            # présente plus tôt qu'une face fréquente.
            probs = torch.sigmoid(self.model(batch) / self.temperature)
            for row in probs:
                out.append([
                    Prediction(cls, float(p))
                    for cls, p in zip(self.meta.classes, row, strict=True)
                    if float(p) >= self.threshold_for(cls, threshold)
                ])
        return out

    @torch.inference_mode()
    def predict_proba(self, image: Image.Image) -> dict:
        """Distribution complète — utile pour le debug et le calibrage des seuils."""
        if not self.available:
            return {}
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x) / self.temperature
        probs = (torch.sigmoid(logits)[0] if self.multilabel
                 else torch.softmax(logits, dim=1)[0])
        return {c: float(p) for c, p in zip(self.meta.classes, probs, strict=True)}
