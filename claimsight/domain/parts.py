"""Agrégation des pièces détectées en constat par pièce.

Le détecteur rend des boîtes par image; un dossier en contient plusieurs, et
la même pièce apparaît souvent sur deux ou trois clichés. Ce module réduit ces
observations à une ligne par pièce — ce qu'attend `damaged_parts[]` du schéma.

Deux règles portent le résultat:

1. **La pire observation gagne**, par gravité ordinale et non par confiance.
   Une même aile vue sous deux angles peut sembler intacte de face et
   franchement enfoncée de trois quarts: c'est la seconde qui fait foi.
2. **Seules les pièces réellement endommagées sont retenues.** Une pièce
   détectée mais saine n'a rien à faire dans un constat de sinistre; elle
   reste dans `raw_detections`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .aggregation import is_real_damage
from .taxonomy import COVERAGE_FACES, UNKNOWN, severity_rank
from .types import Prediction


@dataclass(frozen=True)
class PartObservation:
    """Une pièce vue sur une image, avec le dégât lu sur son propre découpage."""

    part: str
    image: str
    detection_confidence: float
    damage: Prediction
    severity: Prediction
    #: Faces documentées par l'image d'où vient l'observation.
    faces: tuple[str, ...] = field(default_factory=tuple)

    @property
    def severity_rank(self) -> int:
        return severity_rank(self.severity.label)

    @property
    def has_damage(self) -> bool:
        return is_real_damage(self.damage.label)

    def _sort_key(self) -> tuple:
        return (self.severity_rank, int(self.has_damage), self.damage.confidence)

    def confidence(self) -> float:
        """Confiance de la constatation: détection ET dégât doivent tenir.

        On multiplie plutôt qu'on ne moyenne: une pièce mal détectée rend son
        diagnostic de dégât peu crédible, même si le classifieur est sûr de lui.
        """
        if self.damage.label == UNKNOWN:
            return 0.0
        return self.detection_confidence * self.damage.confidence


def aggregate_parts(observations: list[PartObservation]) -> list[dict]:
    """Constat par pièce, trié du plus grave au moins grave.

    Seules les pièces portant un dégât constaté sont retournées.
    """
    by_part: dict[str, list[PartObservation]] = defaultdict(list)
    for obs in observations:
        by_part[obs.part].append(obs)

    entries = []
    for part, group in by_part.items():
        worst = max(group, key=PartObservation._sort_key)
        if not worst.has_damage:
            continue
        faces = {f for obs in group for f in obs.faces}
        entries.append({
            "name": part,
            "damage": worst.damage.label,
            "severity": worst.severity.label,
            "confidence": round(worst.confidence(), 4),
            "views_evidence": sorted(faces, key=COVERAGE_FACES.index),
            "detected_on": sorted({obs.image for obs in group}),
        })

    entries.sort(key=lambda e: (-severity_rank(e["severity"]), -e["confidence"]))
    return entries


def structural_parts_damaged(entries: list[dict], min_severity: str = "severe") -> int:
    """Nombre de pièces atteintes au moins à `min_severity`.

    Sert une heuristique de perte totale plus défendable que le comptage
    d'images graves: ce sont les PIÈCES touchées qui déterminent le coût.
    """
    floor = severity_rank(min_severity)
    return sum(1 for e in entries if severity_rank(e["severity"]) >= floor)
