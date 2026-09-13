"""Contrat de l'API, en Pydantic.

Ces modèles sont la SOURCE DE VÉRITÉ du format de sortie: ils remplacent les
vérifications `isinstance` écrites à la main dans `sanity_check.py`, et FastAPI
en dérive l'OpenAPI — le front récupère donc un contrat typé sans qu'on ait à
le documenter séparément.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..domain.taxonomy import COVERAGE_FACES


class LabelPrediction(BaseModel):
    label: str = Field(description="Classe prédite, ou `unknown` en cas d'abstention")
    confidence: float = Field(ge=0.0, le=1.0)


class DetectedObject(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(min_length=4, max_length=4, description="[x1, y1, x2, y2] en pixels")


class RawDetection(BaseModel):
    image: str
    model_config = ConfigDict(protected_namespaces=())
    model: str = Field(description="Modèle ayant produit les détections")
    objects: list[DetectedObject]


class DamagedPart(BaseModel):
    name: str
    damage: str
    severity: str
    confidence: float = Field(ge=0.0, le=1.0)
    views_evidence: list[str] = Field(description="Faces où la pièce a été constatée")
    detected_on: list[str] = Field(description="Fichiers où la pièce a été détectée")


class InputImage(BaseModel):
    filename: str
    detected_view: str
    view_prediction: LabelPrediction
    damage_prediction: LabelPrediction
    severity_prediction: LabelPrediction
    covered_faces: list[str] = Field(default_factory=list)
    quality_flag: str
    deduplicated: bool
    duplicate_of: str | None = None


class ImageLevelDamage(BaseModel):
    damage: str
    severity: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_image: str | None = None


class ClaimReport(BaseModel):
    """Rapport d'analyse d'un dossier de sinistre."""

    pipeline_version: str
    status: str = Field(description="ok · ok_with_warnings · error")
    vehicle_id: str | None = None
    summary: str

    damaged_parts: list[DamagedPart] = Field(default_factory=list)
    missing_photos: list[str] = Field(
        default_factory=list,
        description=f"Faces non documentées, parmi {list(COVERAGE_FACES)}",
    )
    covered_faces: list[str] = Field(default_factory=list)
    input_images: list[InputImage] = Field(default_factory=list)
    image_level_damage: ImageLevelDamage
    raw_detections: list[RawDetection] = Field(default_factory=list)

    suspected_total_loss: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    duplicates_removed: int = 0

    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    processing_time_ms: int
    inference_time_ms: int | None = None


class ModelStatus(BaseModel):
    loaded: bool
    detail: str | None = None


class Health(BaseModel):
    status: str = Field(description="ready quand au moins l'orientation est disponible")
    pipeline_version: str
    models: dict[str, ModelStatus]


# --------------------------------------------------------------- cycle de vie

class ClaimCreated(BaseModel):
    """Réponse 202 au dépôt d'un dossier."""

    claim_id: str
    status: str = Field(description="queued au moment du dépôt")
    n_images: int


class ClaimStatus(BaseModel):
    """État d'un dossier, interrogé jusqu'à `done` ou `error`."""

    claim_id: str
    status: str = Field(description="queued · processing · done · error")
    error: str | None = None
    created_at: str | None = None
    n_images: int = 0
    result: ClaimReport | None = None


class ClaimSummary(BaseModel):
    claim_id: str
    status: str
    created_at: str | None = None
    n_images: int = 0
