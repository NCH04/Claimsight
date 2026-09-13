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


def compute_multilabel_metrics(
    y_true, y_pred, classes: Sequence[str]
) -> dict:
    """Métriques multi-label. `y_true`/`y_pred` sont des matrices 0/1 (N, C)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out: dict = {
        # La contrainte la plus dure: toutes les faces d'une image correctes.
        "exact_match": float((y_true == y_pred).all(axis=1).mean()),
        "hamming_accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "support": int(len(y_true)),
    }
    # Le F1 par face compte: `left` et `right` sont rares et c'est là que le
    # modèle décroche, ce que la moyenne masque.
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0,
                         labels=list(range(len(classes))))
    for name, value in zip(classes, per_class, strict=True):
        out[f"f1_{name}"] = float(value)
    return out


def multilabel_report(y_true, y_pred, classes: Sequence[str]) -> dict:
    """Détail par face: précision, rappel, F1, support."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = {}
    for i, name in enumerate(classes):
        t, p = y_true[:, i], y_pred[:, i]
        tp = int((t & p).sum()) if t.dtype == bool else int(((t == 1) & (p == 1)).sum())
        fp = int(((t == 0) & (p == 1)).sum())
        fn = int(((t == 1) & (p == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        rows[name] = {
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "support": int((t == 1).sum()),
        }
    return {"per_face": rows}


def format_multilabel(report: dict) -> str:
    lines = [f"{'face':>8s}  {'prec':>6s} {'rappel':>6s} {'F1':>6s} {'n':>6s}"]
    for name, r in report["per_face"].items():
        lines.append(f"{name:>8s}  {r['precision']:6.3f} {r['recall']:6.3f} "
                     f"{r['f1']:6.3f} {r['support']:6d}")
    return "\n".join(lines)


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
