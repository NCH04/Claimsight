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

    #: Multi-label, prédit directement les faces documentées par une photo.
    #: C'est lui qui alimente `missing_photos`; le modèle de vue n'est plus
    #: qu'un complément descriptif.
    coverage: ImageClassifier
    view: ImageClassifier
    damage: ImageClassifier
    severity: ImageClassifier

    @classmethod
    def load(
        cls,
        coverage_checkpoint: str | Path | None = None,
        view_checkpoint: str | Path | None = None,
        damage_checkpoint: str | Path | None = None,
        severity_checkpoint: str | Path | None = None,
        device: str | None = None,
        thresholds: dict | None = None,
    ) -> ModelBundle:
        has_coverage = coverage_checkpoint and Path(coverage_checkpoint).exists()
        has_view = view_checkpoint and Path(view_checkpoint).exists()
        if not (has_coverage or has_view):
            raise FileNotFoundError(
                "Aucun modèle d'orientation disponible.\n"
                f"  couverture : {coverage_checkpoint or '(non fourni)'}\n"
                f"  vue        : {view_checkpoint or '(non fourni)'}\n"
                "Entraînez au moins le modèle de couverture:\n"
                "  python scripts/derive_coverage_labels.py --dataset <dataset_pièces> "
                "--out dataset/coverage_derived.csv\n"
                "  claimsight-train --task coverage --csv_path dataset/coverage_derived.csv "
                "--images_dir <images> --final_fit"
            )

        # Sous le seuil, le classifieur renvoie `unknown` plutôt qu'une
        # étiquette peu fiable: c'est ce qui empêche une vue devinée à 0.18
        # de compter comme couverture photo.
        limits = {**MIN_CONFIDENCE, **(thresholds or {})}
        bundle = cls(
            coverage=ImageClassifier(coverage_checkpoint, device, 0.0, "coverage"),
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
