"""Loading the pipeline's models.

Remplace view_model.py / damage_model.py / severity_model.py, qui étaient
trois classes quasi identiques. Tout passe désormais par
`claimsight.ml.classifier.ImageClassifier`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..ml.classifier import ImageClassifier
from ..ml.detector import PartDetector
from .config import MIN_CONFIDENCE, PARTS_MIN_CONFIDENCE

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """The pipeline's classifiers and detector, loaded once.

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
    #: Détecteur de pièces. Alimente `raw_detections` et, croisé avec le
    #: classifieur de dégâts, `damaged_parts`.
    parts: PartDetector

    @classmethod
    def load(
        cls,
        coverage_checkpoint: str | Path | None = None,
        view_checkpoint: str | Path | None = None,
        damage_checkpoint: str | Path | None = None,
        severity_checkpoint: str | Path | None = None,
        parts_checkpoint: str | Path | None = None,
        device: str | None = None,
        thresholds: dict | None = None,
    ) -> ModelBundle:
        has_coverage = coverage_checkpoint and Path(coverage_checkpoint).exists()
        has_view = view_checkpoint and Path(view_checkpoint).exists()
        if not (has_coverage or has_view):
            raise FileNotFoundError(
                "No coverage model available — the pipeline cannot start.\n"
                f"  coverage : {coverage_checkpoint or '(not provided)'}\n"
                f"  view     : {view_checkpoint or '(not provided)'}\n"
                "\n"
                "Weights are not in the repository. Either download them:\n"
                "  https://github.com/NCH04/Claimsight/releases/latest\n"
                "  (extract into models/)\n"
                "\n"
                "or train the coverage model yourself:\n"
                "  python scripts/derive_coverage_labels.py --dataset <parts_dataset> "
                "--out dataset/coverage_derived.csv\n"
                "  claimsight-train --task coverage --csv_path dataset/coverage_derived.csv "
                "--images_dir <images> --final_fit"
            )

        # Below the threshold the classifier returns `unknown` rather than an
        # unreliable label: that is what stops a face guessed at 0.18 from
        # counting as photo coverage.
        limits = {**MIN_CONFIDENCE, **(thresholds or {})}
        bundle = cls(
            coverage=ImageClassifier(coverage_checkpoint, device, 0.0, "coverage"),
            view=ImageClassifier(view_checkpoint, device, limits["view"], "view"),
            damage=ImageClassifier(damage_checkpoint, device, limits["damage"], "damage"),
            severity=ImageClassifier(severity_checkpoint, device, limits["severity"], "severity"),
            parts=PartDetector(parts_checkpoint, device, PARTS_MIN_CONFIDENCE),
        )
        if not bundle.parts.available:
            LOGGER.warning(
                "Part detector unavailable -> `damaged_parts` and `raw_detections` "
                "will stay empty. Download the weights, or train it with: "
                "claimsight-train-parts"
            )
        for name in ("damage", "severity"):
            if not getattr(bundle, name).available:
                LOGGER.warning(
                    "Model %s unavailable -> `unknown` predictions. Download the "
                    "weights, or train it with: python -m "
                    "claimsight.training.train_classifier --task %s ...",
                    name, name,
                )
        return bundle
