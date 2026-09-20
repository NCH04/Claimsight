"""Entraînement du détecteur de pièces (YOLO / ultralytics).

Le dataset YOLO existait dans scripts/ mais n'était consommé par aucun code:
pas de data.yaml, pas d'entraînement, et `damaged_parts` restait vide dans la
sortie du pipeline. Ce module ferme la boucle.

Prérequis:
    pip install -e ".[yolo]"
    python scripts/remap_yolo_parts.py --src <labels_bruts> --dst dataset/parts_yolo/labels
    python scripts/split_yolo_train_val.py --root dataset/parts_yolo
    python scripts/build_yolo_config.py

    python -m claimsight.training.train_parts_yolo --epochs 100 --model yolov8n.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

LOGGER = logging.getLogger("train_parts")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", default="configs/parts_yolo.yaml",
                    help="data.yaml généré par scripts/build_yolo_config.py")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="Poids de départ (yolov8n/s/m.pt) ou .yaml pour partir de zéro")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="'0', 'cpu', ... (auto par défaut)")
    ap.add_argument("--project", default="outputs/train/parts")
    ap.add_argument("--name", default="yolo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=25)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dépend de l'installation
        raise SystemExit(
            "ultralytics n'est pas installé. Lancez:  pip install -e \".[yolo]\""
        ) from exc

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"{data_path} introuvable. Générez-le d'abord:\n"
            "  python scripts/build_yolo_config.py"
        )

    LOGGER.info("Dataset : %s", data_path)
    LOGGER.info("Model   : %s | epochs=%d imgsz=%d", args.model, args.epochs, args.imgsz)

    # Ultralytics résout un `project` RELATIF sous son propre `runs_dir`, ce qui
    # enterre la sortie dans runs/segment/<project>/. On impose donc un chemin
    # absolu pour que --project désigne bien le dossier demandé.
    project = Path(args.project).resolve()

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project),
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        exist_ok=True,
    )
    metrics = model.val()
    LOGGER.info("mAP50-95=%.4f | mAP50=%.4f", metrics.box.map, metrics.box.map50)
    LOGGER.info("Poids -> %s", project / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
