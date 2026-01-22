import argparse
import os
import time
from typing import Dict, List, Tuple

import torch

from .config import CONFIDENCE_WEIGHTS
from .damage_model import DamageModel
from .io_utils import ensure_parent, list_images, load_image, save_json
from .missing_photos import detect_missing
from .severity_model import SeverityModel
from .summary import make_summary
from .view_model import ViewModel


def _load_view_model(path: str) -> ViewModel:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"View checkpoint not found at {path}. "
            "Train one with src/train_view_kfold.py (torch.save with keys: model, classes, arch, img_size, normalize)."
        )
    return ViewModel(path)


def aggregate_damage(damage_preds: List[Tuple[str, float]]) -> Tuple[str, float]:
    if not damage_preds:
        return "unknown", 0.0
    best = max(damage_preds, key=lambda x: x[1])
    return best


def aggregate_severity(sev_preds: List[Tuple[str, float]]) -> Tuple[str, float]:
    if not sev_preds:
        return "unknown", 0.0
    best = max(sev_preds, key=lambda x: x[1])
    return best


def compute_global_conf(view_preds: List[Dict], damage_conf: float) -> float:
    if view_preds:
        view_avg = sum(p.get("confidence", 0.0) for p in view_preds) / len(view_preds)
    else:
        view_avg = 0.0
    w_view = CONFIDENCE_WEIGHTS["view"]
    w_damage = CONFIDENCE_WEIGHTS["damage"]
    return max(0.0, min(1.0, w_view * view_avg + w_damage * damage_conf))


def run_pipeline(
    images_dir: str,
    out_json: str,
    view_checkpoint: str,
    damage_checkpoint: str = None,
    severity_checkpoint: str = None,
) -> Dict:
    t0 = time.time()
    print(f"[INFO] Listing images in {images_dir}")
    try:
        image_paths = list_images(images_dir)
    except Exception as e:
        result = {
            "pipeline_version": "v1",
            "status": "error",
            "vehicle_id": None,
            "summary": f"Failed to list images: {e}",
            "damaged_parts": [],
            "missing_photos": [],
            "input_images": [],
            "suspected_total_loss": False,
            "confidence": 0.0,
            "raw_detections": [],
            "errors": [str(e)],
            "processing_time_ms": int((time.time() - t0) * 1000),
        }
        save_json(result, out_json)
        return result

    print(f"[INFO] Loading view model from {view_checkpoint}")
    view_model = _load_view_model(view_checkpoint)

    damage_model = DamageModel(damage_checkpoint)
    severity_model = SeverityModel(severity_checkpoint)
    if damage_model.is_dummy:
        print("[WARN] Damage model checkpoint missing -> using dummy predictions (unknown, 0.0)")
    if severity_model.is_dummy:
        print("[WARN] Severity model checkpoint missing -> using dummy predictions (unknown, 0.0)")

    view_preds: List[Dict] = []
    damage_preds: List[Tuple[str, float]] = []
    severity_preds: List[Tuple[str, float]] = []
    input_images: List[Dict] = []
    errors: List[str] = []

    for p in image_paths:
        try:
            img = load_image(p)
        except Exception as e:
            errors.append(f"{p.name}: {e}")
            continue

        view, v_conf = view_model.predict(img)
        dmg, d_conf = damage_model.predict(img)
        sev, s_conf = severity_model.predict(img)

        view_preds.append({"image": p.name, "view": view, "confidence": v_conf})
        damage_preds.append((dmg, d_conf))
        severity_preds.append((sev, s_conf))
        input_images.append({
            "filename": p.name,
            "detected_view": view,  # legacy compatibility with schema
            "view_prediction": {"label": view, "confidence": round(v_conf, 4)},
            "damage_prediction": {"label": dmg, "confidence": round(d_conf, 4)},
            "severity_prediction": {"label": sev, "confidence": round(s_conf, 4)},
            "quality_flag": "ok",
            "deduplicated": False,
        })

    missing = detect_missing([vp["view"] for vp in view_preds])

    agg_damage, damage_conf = aggregate_damage(damage_preds)
    agg_severity, severity_conf = aggregate_severity(severity_preds)

    summary = make_summary(view_preds, agg_damage, agg_severity) if view_preds else "No valid images processed."
    global_conf = compute_global_conf(view_preds, damage_conf) if view_preds else 0.0

    status = "ok"
    if errors and view_preds:
        status = "ok_with_warnings"
    if not view_preds:
        status = "error"

    result = {
        "pipeline_version": "v1",
        "status": status,
        "vehicle_id": None,
        "summary": summary,
        "damaged_parts": [],
        "missing_photos": missing,
        "input_images": input_images,
        "suspected_total_loss": False,
        "confidence": round(global_conf, 4),
        "raw_detections": [],
        "errors": errors,
        "processing_time_ms": int((time.time() - t0) * 1000),
        "image_level_damage": {
            "damage": agg_damage,
            "severity": agg_severity,
            "confidence": round(max(damage_conf, severity_conf), 4),
        },
        "views_detected": view_preds,  # debug/legacy
    }

    print(f"[INFO] Saving results to {out_json}")
    save_json(result, out_json)
    return result


def parse_args():
    ap = argparse.ArgumentParser(description="Run V1 auto damage pipeline (views + image-level damage/severity).")
    ap.add_argument("--images_dir", type=str, required=True, help="Folder containing input images")
    ap.add_argument("--out_json", type=str, default="outputs/result.json", help="Output JSON path")
    ap.add_argument("--view_checkpoint", type=str, default="models/view.pt", help="Path to view classification checkpoint")
    ap.add_argument("--damage_checkpoint", type=str, default=None, help="Optional damage classifier checkpoint")
    ap.add_argument("--severity_checkpoint", type=str, default=None, help="Optional severity classifier checkpoint")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_pipeline(
            images_dir=args.images_dir,
            out_json=args.out_json,
            view_checkpoint=args.view_checkpoint,
            damage_checkpoint=args.damage_checkpoint,
            severity_checkpoint=args.severity_checkpoint,
        )
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        raise SystemExit(1)
