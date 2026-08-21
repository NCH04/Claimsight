"""Contrôle rapide du pipeline V1 sur quelques images.

    python sanity_check.py --images_dir dataset/images_mapped \
        --view_checkpoint models/view.pt --max_images 3
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

from claimsight.pipeline.io_utils import list_images
from claimsight.pipeline.run_pipeline import run_pipeline

LOGGER = logging.getLogger("sanity")

#: Contrat minimal des prédictions par image.
PREDICTION_KEYS = ("view_prediction", "damage_prediction", "severity_prediction")
LIST_KEYS = ("damaged_parts", "missing_photos", "input_images", "raw_detections", "errors")


def expected_keys(schema_path: Path) -> list[str]:
    """Clés attendues, dérivées de l'exemple de référence.

    L'argument --schema était auparavant inerte: on vérifiait seulement que le
    fichier existait. Ici il pilote réellement la validation.
    """
    if not schema_path.exists():
        raise FileNotFoundError(f"Référence de schéma introuvable: {schema_path}")
    reference = json.loads(schema_path.read_text(encoding="utf-8"))
    return sorted(reference.keys())


def validate(result: dict, schema_path: Path) -> list[str]:
    errors: list[str] = []

    for key in expected_keys(schema_path):
        if key not in result:
            errors.append(f"Clé manquante (présente dans le schéma de référence): {key}")

    for key in LIST_KEYS:
        if not isinstance(result.get(key), list):
            errors.append(f"{key} doit être une liste")

    if not isinstance(result.get("image_level_damage"), dict):
        errors.append("image_level_damage doit être un dict")

    for i, item in enumerate(result.get("input_images") or []):
        if not isinstance(item, dict):
            errors.append(f"input_images[{i}] doit être un dict")
            continue
        if "filename" not in item:
            errors.append(f"input_images[{i}] sans filename")
        for key in PREDICTION_KEYS:
            pred = item.get(key)
            if not isinstance(pred, dict) or "label" not in pred or "confidence" not in pred:
                errors.append(f"input_images[{i}].{key} doit avoir label + confidence")
            elif not 0.0 <= pred["confidence"] <= 1.0:
                errors.append(f"input_images[{i}].{key}.confidence hors [0,1]")

    conf = result.get("confidence")
    if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
        errors.append("confidence doit être un nombre dans [0,1]")

    if result.get("status") not in ("ok", "ok_with_warnings", "error"):
        errors.append(f"status inattendu: {result.get('status')!r}")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--view_checkpoint", required=True)
    ap.add_argument("--damage_checkpoint", default=None)
    ap.add_argument("--severity_checkpoint", default=None)
    ap.add_argument("--schema", default="specs/pipeline_schema.json")
    ap.add_argument("--out_json", default="outputs/sanity_result.json")
    ap.add_argument("--max_images", type=int, default=3)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sample = list_images(args.images_dir)[: args.max_images]
    tmp_dir = Path(tempfile.mkdtemp(prefix="sanity_imgs_"))
    try:
        for path in sample:
            shutil.copy2(path, tmp_dir / path.name)

        run_pipeline(
            images_dir=str(tmp_dir),
            out_json=args.out_json,
            view_checkpoint=args.view_checkpoint,
            damage_checkpoint=args.damage_checkpoint,
            severity_checkpoint=args.severity_checkpoint,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    saved = json.loads(Path(args.out_json).read_text(encoding="utf-8"))
    problems = validate(saved, Path(args.schema))

    print(f"[SANITY] status={saved.get('status')} images={len(saved.get('input_images', []))} "
          f"confidence={saved.get('confidence')}")
    for warning in saved.get("warnings", []):
        print(f"[SANITY] avertissement: {warning}")
    for err in saved.get("errors", []):
        print(f"[SANITY] erreur image: {err}")

    if problems:
        print(f"[SANITY] VALIDATION ÉCHOUÉE ({len(problems)}):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("[SANITY] Validation du schéma OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
