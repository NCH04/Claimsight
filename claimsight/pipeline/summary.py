"""Phrase de synthèse d'un dossier.

L'ordre des sources reflète ce qu'un gestionnaire veut lire en premier: la
pièce touchée d'abord, l'orientation seulement si on ne sait pas nommer la
pièce. La vue était la seule source en V0, avant que le détecteur de pièces
n'existe; elle reste en dernier recours.
"""

from __future__ import annotations

#: Au-delà, la phrase devient un inventaire: on cite et on compte le reste.
MAX_PARTS_CITED = 2


def _best_view(view_preds: list[dict]) -> str:
    if not view_preds:
        return "Unknown"
    best = max(view_preds, key=lambda x: x.get("confidence", 0.0))
    return best.get("view", "Unknown")


def _enumerate(names: list[str]) -> str:
    """«a», «a and b», puis «a, b and 3 others»."""
    if len(names) == 1:
        return names[0]
    if len(names) <= MAX_PARTS_CITED:
        return f"{', '.join(names[:-1])} and {names[-1]}"
    rest = len(names) - MAX_PARTS_CITED
    cited = ", ".join(names[:MAX_PARTS_CITED])
    return f"{cited} and {rest} other part{'s' if rest > 1 else ''}"


def _qualifier(damage: str, severity: str) -> str:
    """«moderate dent», ou «dent» seul quand la gravité n'est pas connue."""
    return f"{severity} {damage}" if severity not in ("unknown", "none") else damage


def make_summary(
    view_preds: list[dict],
    damage: str,
    severity: str,
    damaged_parts: list[dict] | None = None,
) -> str:
    if damage == "unknown" and severity == "unknown":
        return "Insufficient evidence to assess damage."
    if damage == "none" and severity == "none":
        return "No visible damage detected."

    # `aggregate_parts` trie déjà par gravité: la première est la pire.
    if damaged_parts:
        names = [p["name"] for p in damaged_parts if p.get("name")]
        if names:
            # Le constat doit décrire les pièces citées, pas le verdict lu sur
            # l'image entière: les deux peuvent diverger, et nommer une pièce
            # avec le dégât d'une autre est pire que ne rien nommer.
            worst = damaged_parts[0]
            subject = _enumerate(names)
            verb = "shows" if len(names) == 1 else "show"
            return (f"{subject.capitalize()} {verb} "
                    f"{_qualifier(worst.get('damage', damage), worst.get('severity', severity))} "
                    f"damage.")

    view = _best_view(view_preds)
    return f"{view.title()} view shows {_qualifier(damage, severity)} damage."
