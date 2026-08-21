"""Types partagés entre le domaine et la couche ML.

Vit dans `domain` (et non dans `ml`) pour que la logique métier —
agrégation, seuils, déduplication — reste importable sans torch, donc
testable sans GPU ni dataset.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    """Une prédiction mono-label avec sa confiance softmax."""

    label: str
    confidence: float

    def as_dict(self, ndigits: int = 4) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, ndigits)}
