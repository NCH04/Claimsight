# Sanity check for the V1 pipeline.
# Usage:
#   python sanity_check.py \
#     --images_dir dataset/images_mapped \
#     --view_checkpoint models/view.pt \
#     --schema specs/pipeline_schema.json \
#     --out_json outputs/sanity_result.json \
#     --max_images 3

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

from src.pipeline.io_utils import list_images
from src.pipeline.run_pipeline import run_pipeline


def validate_schema(result: Dict, schema_path: str) -> List[str]:
    """Validation minimale: présence de clés et types de base."""
    errors = []
    required_keys = [
        "pipeline_version",
        "status",
        "vehicle_id",
        "summary",
        "damaged_parts",
        "missing_photos",
        "input_images",
        "suspected_total_loss",
        "confidence",
        "raw_detections",
        "errors",
        "processing_time_ms",
        "image_level_damage",
    ]
    for k in required_keys:
        if k not in result:
            errors.append(f"Missing key: {k}")

    if not isinstance(result.get("input_images", None), list):
        errors.append("input_images must be a list")
    else:
        for i, item in enumerate(result["input_images"]):
            if not isinstance(item, dict):
                errors.append(f"input_images[{i}] must be a dict")
                continue
            for key in ["filename", "view_prediction", "damage_prediction", "severity_prediction"]:
                if key not in item:
                    errors.append(f"input_images[{i}] missing {key}")
            for pred_key in ["view_prediction", "damage_prediction", "severity_prediction"]:
                pred = item.get(pred_key, {})
                if not isinstance(pred, dict) or "label" not in pred or "confidence" not in pred:
                    errors.append(f"input_images[{i}].{pred_key} must have label/confidence")
    if not isinstance(result.get("missing_photos", None), list):
        errors.append("missing_photos must be a list")
    if not isinstance(result.get("damaged_parts", None), list):
        errors.append("damaged_parts must be a list")
    if not isinstance(result.get("raw_detections", None), list):
        errors.append("raw_detections must be a list")
    if not isinstance(result.get("errors", None), list):
        errors.append("errors must be a list")
    if not isinstance(result.get("image_level_damage", None), dict):
        errors.append("image_level_damage must be a dict")

    conf = result.get("confidence", None)
    if conf is not None and not (isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0):
        errors.append("confidence must be in [0,1]")

    if not Path(schema_path).exists():
        errors.append(f"Schema file not found at {schema_path} (validation skipped)")

    return errors


def main():
    ap = argparse.ArgumentParser(description="Run a quick sanity check on the V1 pipeline output.")
    ap.add_argument("--images_dir", type=str, required=True, help="Folder containing input images")
    ap.add_argument("--view_checkpoint", type=str, required=True, help="View model checkpoint path")
    ap.add_argument("--damage_checkpoint", type=str, default=None, help="Optional damage checkpoint")
    ap.add_argument("--severity_checkpoint", type=str, default=None, help="Optional severity checkpoint")
    ap.add_argument("--schema", type=str, default="specs/pipeline_schema.json", help="Schema path for validation")
    ap.add_argument("--out_json", type=str, default="outputs/sanity_result.json", help="Where to write the pipeline output")
    ap.add_argument("--max_images", type=int, default=3, help="Number of images to test")
    args = ap.parse_args()

    # Sample a few images into a temp dir to speed up the check
    all_imgs = list_images(args.images_dir)
    sample_imgs = all_imgs[: args.max_images]
    tmp_dir = Path(tempfile.mkdtemp(prefix="sanity_imgs_"))
    try:
        for p in sample_imgs:
            shutil.copy2(p, tmp_dir / p.name)

        result = run_pipeline(
            images_dir=str(tmp_dir),
            out_json=args.out_json,
            view_checkpoint=args.view_checkpoint,
            damage_checkpoint=args.damage_checkpoint,
            severity_checkpoint=args.severity_checkpoint,
        )
        with open(args.out_json, "r", encoding="utf-8") as f:
            saved = json.load(f)

        errs = validate_schema(saved, args.schema)
        print(f"[SANITY] status={saved.get('status')} errors_count={len(saved.get('errors', []))}")
        print(f"[SANITY] first input_images: {saved.get('input_images', [])[:1]}")
        if errs:
            print("[SANITY] Schema validation FAILED:")
            for e in errs:
                print(f" - {e}")
        else:
            print("[SANITY] Schema validation OK.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
