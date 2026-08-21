from collections.abc import Iterable

FRONT = {"front", "front-left", "front-right"}
REAR = {"rear", "rear-left", "rear-right"}
LEFT = {"left", "front-left", "rear-left"}
RIGHT = {"right", "front-right", "rear-right"}


def detect_missing(views: Iterable[str]) -> list[str]:
    """Retourne la liste des catégories manquantes parmi front/rear/left/right."""
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

