"""Métriques d'évaluation, y compris le cas ordinal (gravité)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ..domain.taxonomy import severity_rank


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
    ordinal: bool = False,
) -> dict:
    """Métriques d'un fold. `classes` indexe les entiers de y_true/y_pred."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "support": int(len(y_true)),
    }

    if ordinal:
        # Une confusion minor<->severe est bien plus grave qu'un minor<->moderate:
        # l'accuracy seule ne le voit pas.
        ranks_true = np.array([severity_rank(classes[i]) for i in y_true], dtype=float)
        ranks_pred = np.array([severity_rank(classes[i]) for i in y_pred], dtype=float)
        valid = (ranks_true >= 0) & (ranks_pred >= 0)
        if valid.any():
            diff = np.abs(ranks_true[valid] - ranks_pred[valid])
            out["ordinal_mae"] = float(diff.mean())
            out["off_by_one_accuracy"] = float((diff <= 1).mean())

    return out


def full_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
) -> dict:
    """Rapport par classe + matrice de confusion, sérialisables en JSON."""
    labels = list(range(len(classes)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(classes),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(classes),
    }


def aggregate_folds(fold_metrics: list[dict]) -> dict:
    """Moyenne ± écart-type de chaque métrique scalaire sur les folds."""
    if not fold_metrics:
        return {}
    keys = [k for k, v in fold_metrics[0].items() if isinstance(v, (int, float))]
    summary = {}
    for k in keys:
        vals = [float(m[k]) for m in fold_metrics if k in m]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary


def format_confusion(cm: Sequence[Sequence[int]], classes: Sequence[str]) -> str:
    """Matrice de confusion lisible en console."""
    width = max((len(c) for c in classes), default=4)
    width = max(width, 5)
    header = " " * (width + 2) + " ".join(f"{c[:5]:>5s}" for c in classes)
    lines = [header]
    for name, row in zip(classes, cm, strict=True):
        lines.append(f"{name:>{width}s}  " + " ".join(f"{v:5d}" for v in row))
    return "\n".join(lines)
