"""Pipeline V1: vues + dégât/gravité au niveau image.

    python -m claimsight.pipeline.run_pipeline --images_dir dataset/images_mapped \
        --out_json outputs/result.json --view_checkpoint models/view.pt
"""

from __future__ import annotations

import argparse
import logging
import time

from ..domain.aggregation import (
    aggregate_damage,
    aggregate_severity,
    build_findings,
    global_confidence,
    suspect_total_loss,
    worst_finding,
)
from ..domain.dedup import find_duplicates, hash_images
from ..domain.parts import PartObservation, aggregate_parts, structural_parts_damaged
from ..domain.quality import measure as measure_quality
from ..domain.quality import quality_flag
from ..domain.taxonomy import COVERAGE_FACES, UNKNOWN
from .config import (
    CONFIDENCE_WEIGHTS,
    COVERAGE_THRESHOLD,
    DEDUP_HAMMING_THRESHOLD,
    INFERENCE_BATCH_SIZE,
    MIN_CONFIDENCE,
    TOTAL_LOSS_SEVERE_IMAGES,
    TOTAL_LOSS_SEVERE_PARTS,
)
from .io_utils import list_images, load_image, save_json
from .missing_photos import detect_missing, missing_from_faces
from .models import ModelBundle
from .summary import make_summary

LOGGER = logging.getLogger(__name__)

PIPELINE_VERSION = "v1"


def _empty_result(t0: float, message: str, errors: list[str]) -> dict:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "status": "error",
        "vehicle_id": None,
        "summary": message,
        "damaged_parts": [],
        "missing_photos": [],
        "input_images": [],
        "suspected_total_loss": False,
        "confidence": 0.0,
        "raw_detections": [],
        "errors": errors,
        "warnings": [],
        "processing_time_ms": int((time.time() - t0) * 1000),
        "image_level_damage": {"damage": "unknown", "severity": "unknown", "confidence": 0.0},
        "views_detected": [],
    }


def run_pipeline(
    images_dir: str,
    out_json: str | None = None,
    coverage_checkpoint: str | None = "models/coverage.pt",
    view_checkpoint: str | None = "models/view.pt",
    damage_checkpoint: str | None = "models/damage.pt",
    severity_checkpoint: str | None = "models/severity.pt",
    parts_checkpoint: str | None = "models/parts.pt",
    bundle: ModelBundle | None = None,
    recursive: bool = False,
    batch_size: int = INFERENCE_BATCH_SIZE,
    thresholds: dict | None = None,
    dedup_threshold: int = DEDUP_HAMMING_THRESHOLD,
) -> dict:
    """Analyse un dossier d'images et retourne le rapport V1.

    Args:
        bundle: modèles déjà chargés. À privilégier dans un service — sinon
            les trois modèles sont rechargés à chaque appel.

    Ne lève jamais: tout échec est rendu sous forme de résultat `status=error`,
    de sorte qu'un appelant HTTP obtienne toujours un corps exploitable.
    """
    t0 = time.time()
    errors: list[str] = []
    warnings: list[str] = []

    try:
        image_paths = list_images(images_dir, recursive=recursive)
    except Exception as exc:
        LOGGER.error("Listing des images impossible: %s", exc)
        result = _empty_result(t0, f"Listing des images impossible: {exc}", [str(exc)])
        if out_json:
            save_json(result, out_json)
        return result

    try:
        if bundle is None:
            LOGGER.info("Loading models (coverage=%s, view=%s)",
                        coverage_checkpoint, view_checkpoint)
            bundle = ModelBundle.load(
                coverage_checkpoint, view_checkpoint,
                damage_checkpoint, severity_checkpoint, parts_checkpoint,
                thresholds=thresholds,
            )
    except Exception as exc:
        LOGGER.error("Could not load models: %s", exc)
        result = _empty_result(t0, f"Could not load models: {exc}", [str(exc)])
        if out_json:
            save_json(result, out_json)
        return result

    if not bundle.damage.available:
        warnings.append("Model `damage` not trained: predictions are `unknown`.")
    if not bundle.severity.available:
        warnings.append("Model `severity` not trained: predictions are `unknown`.")

    # Décodage: on isole les images illisibles sans faire échouer le dossier.
    images, kept_paths = [], []
    for path in image_paths:
        try:
            images.append(load_image(path))
            kept_paths.append(path)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    if not images:
        result = _empty_result(t0, "No usable image.", errors)
        result["warnings"] = warnings
        if out_json:
            save_json(result, out_json)
        return result

    # Déduplication AVANT agrégation: plusieurs prises du même angle ne
    # doivent ni peser plusieurs fois dans le constat, ni faire croire à une
    # couverture photo plus large qu'elle ne l'est.
    qualities = [measure_quality(img) for img in images]
    duplicate_of = find_duplicates(hash_images(images), threshold=dedup_threshold)
    n_dupes = sum(1 for d in duplicate_of if d is not None)
    if n_dupes:
        LOGGER.info("%d duplicate(s) detected and excluded from aggregation", n_dupes)

    t_infer = time.time()
    coverage_preds = (
        bundle.coverage.predict_labels(images, COVERAGE_THRESHOLD, batch_size)
        if bundle.coverage.available else [[] for _ in images]
    )
    view_preds = bundle.view.predict_batch(images, batch_size)
    damage_preds = bundle.damage.predict_batch(images, batch_size)
    severity_preds = bundle.severity.predict_batch(images, batch_size)
    inference_ms = int((time.time() - t_infer) * 1000)

    # Détection des pièces, puis lecture du dégât sur le DÉCOUPAGE de chaque
    # pièce: c'est ce qui permet de dire « pare-chocs avant enfoncé » plutôt
    # que « il y a un enfoncement quelque part ».
    detections = bundle.parts.detect(images) if bundle.parts.available else [[] for _ in images]
    filenames = [p.name for p in kept_paths]
    input_images = [
        {
            "filename": name,
            "detected_view": view.label,  # v1 schema compatibility
            "view_prediction": view.as_dict(),
            "damage_prediction": damage.as_dict(),
            "severity_prediction": severity.as_dict(),
            # `unknown` signifie ici « sous le seuil de confiance »: la photo
            # est exploitable mais son orientation n'est pas fiable.
            "covered_faces": [f.label for f in faces],
            # Le flou et la sous-exposition priment sur une vue peu sûre:
            # ce sont eux qui justifient de redemander la photo.
            "quality_flag": quality_flag(
                qual,
                view.confidence if view.label != UNKNOWN else 0.0,
                min_confidence=MIN_CONFIDENCE["view"],
            ),
            "quality": qual.as_dict(),
            "deduplicated": dup is not None,
            "duplicate_of": filenames[dup] if dup is not None else None,
        }
        for name, view, damage, severity, dup, faces, qual in zip(
            filenames, view_preds, damage_preds, severity_preds, duplicate_of,
            coverage_preds, qualities, strict=True,
        )
    ]

    # Seuls les originaux alimentent le constat.
    keep = [i for i, dup in enumerate(duplicate_of) if dup is None]
    findings = build_findings(
        [filenames[i] for i in keep],
        [damage_preds[i] for i in keep],
        [severity_preds[i] for i in keep],
    )
    unique_views = [view_preds[i] for i in keep]

    # Observations par pièce, limitées aux images non-doublons.
    observations: list[PartObservation] = []
    for i in keep:
        faces = tuple(f.label for f in coverage_preds[i])
        crops = [d.crop(images[i]) for d in detections[i]]
        crop_damage = bundle.damage.predict_batch(crops, batch_size) if crops else []
        crop_severity = bundle.severity.predict_batch(crops, batch_size) if crops else []
        for det, dmg, sev in zip(detections[i], crop_damage, crop_severity, strict=True):
            observations.append(PartObservation(
                part=det.label, image=filenames[i],
                detection_confidence=det.confidence,
                damage=dmg, severity=sev, faces=faces,
            ))

    damaged_parts = aggregate_parts(observations)
    raw_detections = [
        {"image": filenames[i], "model": "yolo-parts-v1",
         "objects": [d.as_dict() for d in detections[i]]}
        for i in keep if detections[i]
    ]

    agg_damage = aggregate_damage(findings)
    agg_severity = aggregate_severity(findings)
    worst = worst_finding(findings)

    # Le modèle de couverture prédit directement les faces: une photo en
    # diagonale en documente deux, ce qu'une classe de vue exclusive rendait
    # impossible à exprimer. À défaut, on retombe sur les vues.
    if bundle.coverage.available:
        covered = {f.label for i in keep for f in coverage_preds[i]}
        missing = missing_from_faces(covered)
    else:
        covered = set()
        missing = detect_missing(p.label for p in unique_views)

    legacy_views = [
        {"image": filenames[i], "view": view_preds[i].label, "confidence": view_preds[i].confidence}
        for i in keep
    ]

    result = {
        "pipeline_version": PIPELINE_VERSION,
        "status": "ok_with_warnings" if (errors or warnings) else "ok",
        "vehicle_id": None,
        "summary": make_summary(legacy_views, agg_damage.label, agg_severity.label,
                                damaged_parts),
        "damaged_parts": damaged_parts,
        "missing_photos": missing,
        "input_images": input_images,
        "suspected_total_loss": (
            structural_parts_damaged(damaged_parts) >= TOTAL_LOSS_SEVERE_PARTS
            if bundle.parts.available
            else suspect_total_loss(findings, TOTAL_LOSS_SEVERE_IMAGES)
        ),
        "covered_faces": sorted(covered, key=COVERAGE_FACES.index) if covered else [],
        "confidence": round(
            global_confidence(
                ([p.confidence for p in unique_views] if bundle.view.available
                 else [max((f.confidence for f in coverage_preds[i]), default=0.0) for i in keep]),
                agg_damage.confidence if bundle.damage.available else None,
                CONFIDENCE_WEIGHTS,
            ),
            4,
        ),
        "raw_detections": raw_detections,
        "errors": errors,
        "warnings": warnings,
        "processing_time_ms": int((time.time() - t0) * 1000),
        "inference_time_ms": inference_ms,
        "duplicates_removed": n_dupes,
        "image_level_damage": {
            "damage": agg_damage.label,
            "severity": agg_severity.label,
            # Confiance de la constatation retenue, bornée par son maillon le
            # plus faible — et non un max() entre deux modèles distincts
            # portant potentiellement sur deux images différentes.
            "confidence": round(worst.confidence(), 4) if worst else 0.0,
            "evidence_image": worst.filename if worst else None,
        },
        "views_detected": legacy_views,  # debug/legacy
    }

    if out_json:
        LOGGER.info("Result -> %s", out_json)
        save_json(result, out_json)
    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="V1 pipeline (coverage + image-level damage/severity).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_json", default="outputs/result.json")
    ap.add_argument("--coverage_checkpoint", default="models/coverage.pt",
                    help="Multi-label photo-coverage model (feeds missing_photos)")
    ap.add_argument("--view_checkpoint", default="models/view.pt",
                    help="View model (optional, descriptive only)")
    ap.add_argument("--damage_checkpoint", default="models/damage.pt")
    ap.add_argument("--severity_checkpoint", default="models/severity.pt")
    ap.add_argument("--parts_checkpoint", default="models/parts.pt",
                    help="YOLO part detector (feeds raw_detections and damaged_parts)")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--batch_size", type=int, default=INFERENCE_BATCH_SIZE)
    ap.add_argument("--min_view_confidence", type=float, default=None,
                    help="Sous ce seuil la vue devient `unknown` et ne compte plus comme couverture photo")
    ap.add_argument("--min_damage_confidence", type=float, default=None)
    ap.add_argument("--min_severity_confidence", type=float, default=None)
    ap.add_argument("--dedup_threshold", type=int, default=DEDUP_HAMMING_THRESHOLD,
                    help="Distance de Hamming max entre dHash pour un doublon (0 = identiques stricts)")
    ap.add_argument("--log_level", default="INFO")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(message)s")
    thresholds = {
        name: value
        for name, value in (
            ("view", args.min_view_confidence),
            ("damage", args.min_damage_confidence),
            ("severity", args.min_severity_confidence),
        )
        if value is not None
    }
    result = run_pipeline(
        images_dir=args.images_dir,
        out_json=args.out_json,
        coverage_checkpoint=args.coverage_checkpoint,
        view_checkpoint=args.view_checkpoint,
        damage_checkpoint=args.damage_checkpoint,
        severity_checkpoint=args.severity_checkpoint,
        parts_checkpoint=args.parts_checkpoint,
        recursive=args.recursive,
        batch_size=args.batch_size,
        thresholds=thresholds or None,
        dedup_threshold=args.dedup_threshold,
    )
    for warning in result.get("warnings", []):
        LOGGER.warning(warning)
    for err in result.get("errors", []):
        LOGGER.error(err)
    return 0 if result["status"] != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
