"""Chargement des modèles du pipeline.

Remplace view_model.py / damage_model.py / severity_model.py, qui étaient
trois classes quasi identiques. Tout passe désormais par
`claimsight.ml.classifier.ImageClassifier`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..ml.classifier import ImageClassifier
from .config import MIN_CONFIDENCE

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Les trois classifieurs du pipeline, chargés une seule fois.

    Destiné à être instancié au démarrage d'un service (et non par requête):
    construire un ResNet + charger son state_dict coûte plusieurs centaines de
    millisecondes.
    """

    view: ImageClassifier
    damage: ImageClassifier
    severity: ImageClassifier

    @classmethod
    def load(
        cls,
        view_checkpoint: str | Path,
        damage_checkpoint: str | Path | None = None,
        severity_checkpoint: str | Path | None = None,
        device: str | None = None,
        thresholds: dict | None = None,
    ) -> ModelBundle:
        if not Path(view_checkpoint).exists():
            raise FileNotFoundError(
                f"Checkpoint de vue introuvable: {view_checkpoint}\n"
                "Entraînez-le avec:\n"
                "  python -m claimsight.training.train_classifier --task view "
                "--csv_path dataset/labels.csv --images_dir dataset/images_mapped --final_fit"
            )

        # Sous le seuil, le classifieur renvoie `unknown` plutôt qu'une
        # étiquette peu fiable: c'est ce qui empêche une vue devinée à 0.18
        # de compter comme couverture photo.
        limits = {**MIN_CONFIDENCE, **(thresholds or {})}
        bundle = cls(
            view=ImageClassifier(view_checkpoint, device, limits["view"], "view"),
            damage=ImageClassifier(damage_checkpoint, device, limits["damage"], "damage"),
            severity=ImageClassifier(severity_checkpoint, device, limits["severity"], "severity"),
        )
        for name in ("damage", "severity"):
            if not getattr(bundle, name).available:
                LOGGER.warning(
                    "Modèle %s indisponible -> prédictions `unknown`. "
                    "Entraînez-le avec: python -m claimsight.training.train_classifier --task %s ...",
                    name, name,
                )
        return bundle
