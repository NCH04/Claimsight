"""Source de vérité unique des taxonomies V1.

Tout le reste du projet (entraînement, inférence, génération du data.yaml YOLO,
specs) doit importer d'ici. Aucune liste de classes ne doit être redéfinie
ailleurs: c'est ce qui a causé la dérive entre `specs/supported_parts_v1.md`
et `scripts/clean_yolo_parts.py`.

Ce module ne dépend que de la stdlib: il est importable sans torch.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Vues (classification image-level)
# ---------------------------------------------------------------------------

VIEW_CLASSES: tuple[str, ...] = (
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
)

#: Vues qui décrivent une orientation exploitable pour la couverture photo.
DIRECTIONAL_VIEWS: tuple[str, ...] = VIEW_CLASSES[:8]

#: Effet d'un flip horizontal sur le label de vue.
#: Les vues non-directionnelles se mappent sur elles-mêmes: le flip reste une
#: augmentation valide pour elles (l'ancien code ne les augmentait pas du tout).
VIEW_MIRROR_MAP: Mapping[str, str] = {
    "front": "front",
    "rear": "rear",
    "left": "right",
    "right": "left",
    "front-left": "front-right",
    "front-right": "front-left",
    "rear-left": "rear-right",
    "rear-right": "rear-left",
    "closeup": "closeup",
    "out_of_scope": "out_of_scope",
}

# ---------------------------------------------------------------------------
# Couverture photo (multi-label)
# ---------------------------------------------------------------------------

#: Les quatre faces d'un véhicule qu'un dossier doit documenter.
#: `detect_missing` ne raisonne que sur celles-ci: les dix VIEW_CLASSES s'y
#: rabattaient déjà, une diagonale comptant pour deux faces. Les prédire
#: directement, en multi-label, supprime cette indirection — et rend le
#: problème plus facile à annoter comme à apprendre.
COVERAGE_FACES: tuple[str, ...] = ("front", "rear", "left", "right")

#: Faces couvertes par chacune des vues historiques. Sert à convertir un jeu
#: d'annotations VIEW_CLASSES existant vers la tâche `coverage`.
VIEW_TO_FACES: Mapping[str, tuple[str, ...]] = {
    "front": ("front",),
    "rear": ("rear",),
    "left": ("left",),
    "right": ("right",),
    "front-left": ("front", "left"),
    "front-right": ("front", "right"),
    "rear-left": ("rear", "left"),
    "rear-right": ("rear", "right"),
    "closeup": (),
    "out_of_scope": (),
}

#: Effet d'un flip horizontal sur un vecteur de couverture.
FACE_MIRROR: Mapping[str, str] = {
    "front": "front", "rear": "rear", "left": "right", "right": "left",
}


def faces_of_view(view: str) -> tuple[str, ...]:
    """Faces documentées par une vue de l'ancienne taxonomie."""
    return VIEW_TO_FACES.get(view, ())


def mirror_faces(faces: Iterable[str]) -> tuple[str, ...]:
    """Faces après flip horizontal (gauche et droite permutées)."""
    return tuple(FACE_MIRROR.get(f, f) for f in faces)


# ---------------------------------------------------------------------------
# Dégâts
# ---------------------------------------------------------------------------

#: Types de dégâts annotés (cf. specs/annotation_guidelines.md §2.3).
DAMAGE_LABELS: tuple[str, ...] = (
    "scratch",
    "dent",
    "crack",
    "broken_glass",
    "deformation_impact",
    "missing_part",
)

#: Classes du classifieur de dégâts = labels annotés + l'absence de dégât.
#: `unknown` n'est PAS une classe apprise: c'est la valeur retournée quand
#: aucun modèle n'est disponible ou que la confiance est sous le seuil.
DAMAGE_CLASSES: tuple[str, ...] = ("none",) + DAMAGE_LABELS

# ---------------------------------------------------------------------------
# Gravité (ordinale)
# ---------------------------------------------------------------------------

SEVERITY_CLASSES: tuple[str, ...] = ("none", "minor", "moderate", "severe")

#: Rang ordinal: sert à agréger par gravité MAX et non par confiance max.
SEVERITY_ORDER: Mapping[str, int] = {
    "none": 0,
    "minor": 1,
    "moderate": 2,
    "severe": 3,
}

#: Valeur sentinelle commune, hors vocabulaire appris.
UNKNOWN = "unknown"


def severity_rank(label: str) -> int:
    """Rang ordinal d'une gravité; -1 pour `unknown` / label inconnu."""
    return SEVERITY_ORDER.get(label, -1)


# ---------------------------------------------------------------------------
# Pièces (détection YOLO)
# ---------------------------------------------------------------------------

#: Liste officielle V1, alignée sur specs/supported_parts_v1.md.
#: L'ORDRE EST CONTRACTUEL: l'index dans ce tuple est l'id de classe YOLO.
#: Ne jamais réordonner sans réentraîner et régénérer configs/parts_yolo.yaml.
PART_CLASSES: tuple[str, ...] = (
    "front bumper",
    "rear bumper",
    "hood",
    "trunk",
    "windshield",
    "front left door",
    "front right door",
    "rear left door",
    "rear right door",
    "left fender",
    "right fender",
    "rear left fender",
    "rear right fender",
    "headlights",
    "taillights",
)

#: Exclues de la V1 par décision produit (specs/supported_parts_v1.md §Removed).
#: Conservées ici pour que le remapping du dataset source puisse les ignorer
#: explicitement plutôt que silencieusement.
EXCLUDED_PARTS_V1: tuple[str, ...] = (
    "wheel/rim",
    "side door",
    "rear windshield",
)

PART_TO_ID: Mapping[str, int] = {name: i for i, name in enumerate(PART_CLASSES)}
ID_TO_PART: Mapping[int, str] = {i: name for i, name in enumerate(PART_CLASSES)}


def as_index_map(classes: Sequence[str]) -> dict[str, int]:
    """Construit un mapping label -> index stable pour une liste de classes."""
    return {label: i for i, label in enumerate(classes)}
