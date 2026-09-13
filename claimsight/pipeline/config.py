"""Configuration du pipeline.

Les taxonomies vivent dans `claimsight.domain.taxonomy` (source de vérité unique).
Ce module ne fait que les ré-exporter et porter les réglages d'inférence.
"""

from __future__ import annotations

from ..domain.taxonomy import (
    DAMAGE_CLASSES,
    SEVERITY_CLASSES,
    UNKNOWN,
    VIEW_CLASSES,
)
from ..ml.transforms import IMAGENET_MEAN, IMAGENET_STD
from .io_utils import IMAGE_EXTS

# Conservés sous leurs anciens noms pour ne pas casser le code existant.
DEFAULT_DAMAGE_CLASSES: list[str] = list(DAMAGE_CLASSES) + [UNKNOWN]
DEFAULT_SEVERITY_CLASSES: list[str] = list(SEVERITY_CLASSES) + [UNKNOWN]

#: Pondération de la confiance globale. Renormalisée sur les seules
#: composantes disponibles (cf. domain.aggregation.global_confidence): si le
#: modèle de dégât manque, la vue porte 100% du score au lieu d'être plafonnée.
CONFIDENCE_WEIGHTS: dict[str, float] = {"view": 0.5, "damage": 0.5}

#: Seuils d'abstention: sous ce seuil, la prédiction devient `unknown`.
#: Point important: une vue `unknown` ne compte plus comme couverture photo.
#: Une vue prédite `front` à 0.18 ne doit pas faire croire que la photo avant
#: a été fournie. 0.0 = désactivé; à calibrer sur les courbes de confiance.
MIN_CONFIDENCE: dict[str, float] = {"view": 0.0, "damage": 0.0, "severity": 0.0}

#: Seuil sigmoïde du modèle de couverture: au-delà, la face est considérée
#: documentée. Plus il est haut, plus `missing_photos` est exigeant.
COVERAGE_THRESHOLD = 0.5

#: Confiance minimale d'une détection de pièce retenue.
PARTS_MIN_CONFIDENCE = 0.40

#: Nombre de PIÈCES gravement atteintes déclenchant la suspicion de perte
#: totale. Plus défendable que le comptage d'images graves: c'est la pièce
#: touchée qui détermine le coût de réparation.
TOTAL_LOSS_SEVERE_PARTS = 3

#: Distance de Hamming maximale entre deux dHash pour déclarer un doublon.
DEDUP_HAMMING_THRESHOLD = 5

#: Nombre d'images `severe` distinctes déclenchant la suspicion de perte totale.
TOTAL_LOSS_SEVERE_IMAGES = 3

#: Taille de lot pour l'inférence.
INFERENCE_BATCH_SIZE = 16

__all__ = [
    "VIEW_CLASSES",
    "DEFAULT_DAMAGE_CLASSES",
    "DEFAULT_SEVERITY_CLASSES",
    "CONFIDENCE_WEIGHTS",
    "MIN_CONFIDENCE",
    "COVERAGE_THRESHOLD",
    "DEDUP_HAMMING_THRESHOLD",
    "TOTAL_LOSS_SEVERE_IMAGES",
    "TOTAL_LOSS_SEVERE_PARTS",
    "PARTS_MIN_CONFIDENCE",
    "INFERENCE_BATCH_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMAGE_EXTS",
]
