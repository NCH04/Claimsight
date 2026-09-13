"""Définition déclarative des tâches de classification.

Ajouter un modèle = ajouter une entrée dans TASKS. Le trainer
(`claimsight.training.train_classifier`) est agnostique de la tâche.

Contrat du CSV d'entrée, commun à toutes les tâches:
    image        (requis)  chemin ou nom de fichier de l'image
    <label_col>  (requis)  le label de la tâche (view / damage / severity)
    vehicle_id   (optionnel mais FORTEMENT recommandé)
                           identifiant du véhicule/sinistre. Sans lui, deux
                           photos du même véhicule peuvent tomber de part et
                           d'autre du split -> métriques surévaluées.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..domain.taxonomy import (
    COVERAGE_FACES,
    DAMAGE_CLASSES,
    FACE_MIRROR,
    SEVERITY_CLASSES,
    VIEW_CLASSES,
    VIEW_MIRROR_MAP,
)


@dataclass(frozen=True)
class TaskConfig:
    """Tout ce qui distingue une tâche de classification d'une autre."""

    name: str
    label_column: str
    classes: tuple[str, ...]
    #: Labels présents dans le CSV mais exclus de l'entraînement.
    drop_labels: tuple[str, ...] = ()
    #: Labels exclus seulement si --drop-optional est passé.
    optional_drop_labels: tuple[str, ...] = ()
    #: Effet d'un flip horizontal sur le label. None = le flip préserve le
    #: label (vrai pour damage/severity: un rayure reste une rayure).
    mirror_map: Mapping[str, str] | None = None
    #: La tâche est-elle ordinale (gravité) ? Active des métriques dédiées.
    ordinal: bool = False
    #: Multi-label (plusieurs classes vraies simultanément) plutôt que
    #: mono-label exclusif. Bascule la perte en BCE et la sortie en sigmoïde.
    multilabel: bool = False
    default_backbone: str = "resnet34"
    default_img_size: int = 256
    default_resize_mode: str = "center_crop"
    description: str = ""

    def mirrored_label(self, label: str) -> str:
        """Label après flip horizontal."""
        if self.mirror_map is None:
            return label
        return self.mirror_map.get(label, label)

    def can_mirror(self, label: str) -> bool:
        """Le flip est-il une augmentation valide pour ce label ?"""
        return self.mirror_map is None or label in self.mirror_map


TASKS: dict[str, TaskConfig] = {
    "coverage": TaskConfig(
        name="coverage",
        label_column="",  # multi-label: les classes SONT les colonnes du CSV
        classes=COVERAGE_FACES,
        multilabel=True,
        mirror_map=FACE_MIRROR,
        default_backbone="resnet34",
        default_img_size=256,
        # La couverture se lit sur le véhicule entier: ne rien rogner.
        default_resize_mode="pad_square",
        description="Faces du véhicule documentées par une photo (multi-label).",
    ),
    "view": TaskConfig(
        name="view",
        label_column="view",
        classes=VIEW_CLASSES,
        # Un closeup n'a pas d'orientation exploitable: il pollue la tâche.
        drop_labels=("closeup",),
        optional_drop_labels=("out_of_scope",),
        mirror_map=VIEW_MIRROR_MAP,
        default_backbone="resnet34",
        default_img_size=256,
        # La vue est portée par le véhicule entier: ne rien rogner.
        default_resize_mode="pad_square",
        description="Orientation de la prise de vue (front/rear/left/right/diagonales).",
    ),
    "damage": TaskConfig(
        name="damage",
        label_column="damage",
        classes=DAMAGE_CLASSES,
        mirror_map=None,  # un dégât miroité reste le même dégât
        default_backbone="resnet34",
        default_img_size=256,
        default_resize_mode="center_crop",
        description="Type de dégât dominant sur l'image.",
    ),
    "severity": TaskConfig(
        name="severity",
        label_column="severity",
        classes=SEVERITY_CLASSES,
        mirror_map=None,
        ordinal=True,
        default_backbone="resnet34",
        default_img_size=256,
        default_resize_mode="center_crop",
        description="Gravité du dégât dominant (ordinale: none < minor < moderate < severe).",
    ),
}


def get_task(name: str) -> TaskConfig:
    if name not in TASKS:
        raise ValueError(f"Tâche inconnue: {name!r}. Disponibles: {', '.join(TASKS)}")
    return TASKS[name]
