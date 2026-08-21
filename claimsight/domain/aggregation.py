"""Agrégation d'un dossier de photos en un constat unique.

Bug corrigé ici — l'ancienne version prenait l'argmax de CONFIANCE:

    best = max(damage_preds, key=lambda x: x[1])

La confiance n'est pas une magnitude de dégât. Un `none` prédit à 0.99
l'emportait sur un `dent` à 0.62, donc un véhicule accidenté pouvait
ressortir « aucun dégât » — le pire faux négatif possible pour de
l'assurance. De plus dégât et gravité étaient agrégés indépendamment: le
rapport pouvait combiner le type de dégât d'une photo avec la gravité d'une
autre.

Nouvelle règle: on classe les images par GRAVITÉ ORDINALE et on retient
« la pire constatation », dont le type de dégât et la gravité proviennent
alors de la MÊME image.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .taxonomy import UNKNOWN, severity_rank
from .types import Prediction

NONE = "none"

#: Nombre d'images distinctes montrant `severe` au-delà duquel on suspecte
#: une perte totale. Heuristique V1, à calibrer sur des dossiers réels.
TOTAL_LOSS_SEVERE_IMAGES = 3


def is_real_damage(label: str) -> bool:
    """Un dégât constaté, par opposition à `none` (rien) et `unknown` (on ne sait pas)."""
    return label not in (NONE, UNKNOWN)


@dataclass(frozen=True)
class Finding:
    """Constatation portée par une image: son dégât et sa gravité."""

    index: int
    filename: str
    damage: Prediction
    severity: Prediction

    @property
    def severity_rank(self) -> int:
        return severity_rank(self.severity.label)

    @property
    def has_damage(self) -> bool:
        return is_real_damage(self.damage.label)

    def _sort_key(self) -> tuple:
        # 1. gravité ordinale (unknown = -1, donc classé sous `none`)
        # 2. un dégât constaté prime sur `none` à gravité égale
        # 3. confiance, en dernier recours seulement
        return (self.severity_rank, int(self.has_damage), self.damage.confidence)

    def confidence(self) -> float:
        """Confiance de la constatation, bornée par son maillon le plus faible.

        Les composantes issues d'un modèle indisponible (`unknown`) sont
        ignorées plutôt que comptées comme 0.0, ce qui écraserait le score.
        """
        parts = [
            p.confidence
            for p in (self.damage, self.severity)
            if p.label != UNKNOWN
        ]
        return min(parts) if parts else 0.0


def build_findings(
    filenames: Sequence[str],
    damage_preds: Sequence[Prediction],
    severity_preds: Sequence[Prediction],
) -> list[Finding]:
    return [
        Finding(index=i, filename=name, damage=dmg, severity=sev)
        for i, (name, dmg, sev) in enumerate(
            zip(filenames, damage_preds, severity_preds, strict=True)
        )
    ]


def worst_finding(findings: Sequence[Finding]) -> Finding | None:
    """La constatation la plus grave du dossier, ou None si aucune image."""
    return max(findings, key=Finding._sort_key) if findings else None


def aggregate_damage(findings: Sequence[Finding]) -> Prediction:
    """Type de dégât retenu pour le dossier.

    Un dégât constaté sur une seule photo l'emporte sur n'importe quel nombre
    de `none`: on ne peut pas « voter » l'absence d'un dommage qui est visible
    quelque part.
    """
    damaged = [f for f in findings if f.has_damage]
    if damaged:
        return max(damaged, key=Finding._sort_key).damage

    known = [f.damage for f in findings if f.damage.label != UNKNOWN]
    if known:
        return max(known, key=lambda p: p.confidence)  # tous `none`
    return Prediction(UNKNOWN, 0.0)


def aggregate_severity(findings: Sequence[Finding]) -> Prediction:
    """Gravité retenue = la PIRE observée, pas la plus confiante."""
    known = [f for f in findings if f.severity_rank >= 0]
    if not known:
        return Prediction(UNKNOWN, 0.0)
    return max(known, key=lambda f: (f.severity_rank, f.severity.confidence)).severity


def suspect_total_loss(
    findings: Sequence[Finding],
    min_severe_images: int = TOTAL_LOSS_SEVERE_IMAGES,
) -> bool:
    """Heuristique de perte totale: plusieurs angles distincts en `severe`."""
    severe = sum(1 for f in findings if f.severity.label == "severe")
    return severe >= min_severe_images


def global_confidence(
    view_confidences: Sequence[float],
    damage_confidence: float | None,
    weights: dict,
) -> float:
    """Confiance globale, renormalisée sur les composantes DISPONIBLES.

    L'ancienne formule était `0.5 * vue + 0.5 * dégât`. Le modèle de dégât
    n'étant pas entraîné, sa confiance valait 0.0 et le score global était
    mécaniquement plafonné à 0.5 — même avec des vues parfaites. On ne
    pondère désormais que ce qui existe réellement.
    """
    components: dict = {}
    if view_confidences:
        components["view"] = sum(view_confidences) / len(view_confidences)
    if damage_confidence is not None:
        components["damage"] = damage_confidence

    total_weight = sum(weights.get(name, 0.0) for name in components)
    if not components or total_weight <= 0:
        return 0.0

    score = sum(weights.get(name, 0.0) * value for name, value in components.items())
    return max(0.0, min(1.0, score / total_weight))
