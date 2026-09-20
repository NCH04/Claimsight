from collections.abc import Iterable

from ..domain.taxonomy import COVERAGE_FACES

FRONT = {"front", "front-left", "front-right"}
REAR = {"rear", "rear-left", "rear-right"}
LEFT = {"left", "front-left", "rear-left"}
RIGHT = {"right", "front-right", "rear-right"}


def detect_missing(views: Iterable[str]) -> list[str]:
    """The missing categories among front/rear/left/right."""
    views_set = set(views)
    missing = []
    if not (views_set & FRONT):
        missing.append("front")
    if not (views_set & REAR):
        missing.append("rear")
    if not (views_set & LEFT):
        missing.append("left")
    if not (views_set & RIGHT):
        missing.append("right")
    return missing



def missing_from_faces(covered: Iterable[str]) -> list[str]:
    """Faces with no photo, given the set of covered faces.

    Chemin direct utilisé quand le modèle de couverture (multi-label) est
    disponible: il prédit déjà les faces, sans passer par les dix vues.
    """
    seen = set(covered)
    return [face for face in COVERAGE_FACES if face not in seen]
