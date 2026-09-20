"""Calibration des confiances d'un checkpoint.

Un réseau entraîné est presque toujours SUR-CONFIANT: il annonce 0.98 là où il
a raison 85 % du temps. Tant que ce n'est pas corrigé, un seuil d'abstention
n'a aucun sens — c'est pourquoi `MIN_CONFIDENCE` était resté à 0.0.

Deux opérations, l'une après l'autre:

1. **Temperature scaling** — une seule température scalaire T divise les
   logits, optimisée par LBFGS sur la perte du jeu de validation. L'accuracy
   est strictement inchangée (T > 0 ne réordonne rien), seules les confiances
   deviennent interprétables.
2. **Seuils par classe** (multi-label) — après calibration, on cherche le
   seuil qui maximise le F1 de CHAQUE face. `left` et `right` étant rares,
   leur seuil optimal est plus bas que 0.5; l'imposer uniformément coûterait
   du rappel là où on peut le moins se le permettre.

    python -m claimsight.training.calibrate --checkpoint models/coverage.pt \\
        --csv_path dataset/coverage_derived.csv --images_dir <images>

**Hors-fold.** `--final_fit` entraîne sur 100 % des données: il ne reste alors
aucune image que le modèle final n'ait vue, et calibrer sur du déjà-vu donne
une température trop proche de 1. `--oof_from` contourne le problème sans
sacrifier de données: on rejoue le découpage en folds, chaque checkpoint de
fold prédit SON fold de validation — qu'il n'a jamais vu — et la température
est ajustée sur la concaténation, qui couvre tout le jeu proprement.

    python -m claimsight.training.calibrate --checkpoint models/damage.pt \\
        --csv_path dataset/damage_mixed.csv --images_dir / \\
        --oof_from outputs/train/damage_mixed --group_column photo_id --write
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..ml.backbones import build_backbone
from ..ml.checkpoint import load_checkpoint
from ..ml.transforms import build_eval_transform
from .dataset import ClassificationDataset
from .tasks import get_task

LOGGER = logging.getLogger("calibrate")


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits bruts du jeu de validation.

    `no_grad` et non `inference_mode`: la température est ensuite optimisée par
    LBFGS sur ces tenseurs, et autograd refuse les tenseurs d'inference_mode.
    """
    model.eval()
    logits, targets = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).detach().cpu())
        targets.append(y.detach().cpu())
    return torch.cat(logits), torch.cat(targets)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, multilabel: bool) -> float:
    """Température optimale au sens de la perte du jeu de validation."""
    log_t = torch.zeros(1, requires_grad=True)  # on optimise log(T) pour garder T > 0
    criterion = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / log_t.exp(), targets.float() if multilabel else targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def expected_calibration_error(probs: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """ECE: écart moyen entre confiance annoncée et exactitude observée."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    error, n = 0.0, len(probs)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (probs > lo) & (probs <= hi)
        if mask.sum():
            error += mask.sum() / n * abs(correct[mask].mean() - probs[mask].mean())
    return float(error)


def best_thresholds(probs: np.ndarray, targets: np.ndarray, classes: list[str]) -> dict[str, float]:
    """Seuil maximisant le F1, classe par classe."""
    grid = np.arange(0.05, 0.96, 0.01)
    out = {}
    for i, name in enumerate(classes):
        t, p = targets[:, i], probs[:, i]
        scores = []
        for thr in grid:
            pred = p >= thr
            tp = float((pred & (t == 1)).sum())
            fp = float((pred & (t == 0)).sum())
            fn = float((~pred & (t == 1)).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            scores.append(f1)
        out[name] = float(grid[int(np.argmax(scores))])
    return out


def collect_oof_logits(args, task, meta, device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Logits hors-fold: chaque checkpoint de fold ne voit que SON fold.

    Rejoue exactement le découpage de l'entraînement — même splitter, mêmes
    strates, mêmes groupes, même graine — donc chaque ligne est prédite par le
    seul modèle qui ne l'a pas eue en apprentissage. La concaténation couvre
    tout le jeu sans qu'aucune prédiction ne soit contaminée.
    """
    from .train_classifier import _stratify_key, build_splitter, load_dataframe

    df, labels, group_col = load_dataframe(args, task)
    if list(labels) != list(meta.classes):
        raise SystemExit(
            f"Classes du checkpoint {list(meta.classes)} != classes du CSV {list(labels)}.\n"
            "Le CSV n'est pas celui qui a produit ces folds."
        )

    splitter, grouped = build_splitter(args, group_col)
    if args.group_column and not grouped:
        raise SystemExit(f"Colonne de groupe {args.group_column!r} absente: "
                         "les folds ne seraient pas ceux de l'entraînement.")

    label2idx = {c: i for i, c in enumerate(meta.classes)}
    transform = build_eval_transform(meta.img_size, meta.resize_mode, meta.normalize)
    folds_dir = Path(args.oof_from)

    all_logits, all_targets, n_folds = [], [], 0
    strata = _stratify_key(df, task)
    groups = df[group_col] if grouped else None
    for fold, (_, va_idx) in enumerate(splitter.split(df, strata, groups)):
        ckpt = folds_dir / f"{task.name}_fold{fold}.pt"
        if not ckpt.exists():
            raise SystemExit(f"Checkpoint de fold manquant: {ckpt}")
        state, fold_meta = load_checkpoint(ckpt)
        model = build_backbone(fold_meta.arch, len(fold_meta.classes), pretrained=False)
        model.load_state_dict(state)
        model.to(device).eval()

        dataset = ClassificationDataset(df.iloc[va_idx], args.images_dir, label2idx,
                                        task, transform, train=False)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        logits, targets = collect_logits(model, loader, device)
        LOGGER.info("  fold %d : %d images hors-fold", fold + 1, len(targets))
        all_logits.append(logits)
        all_targets.append(targets)
        n_folds += 1

    return torch.cat(all_logits), torch.cat(all_targets), n_folds


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--csv_path", required=True, help="Jeu de validation, NON vu à l'entraînement")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--task", default=None, help="Défaut: la tâche inscrite dans le checkpoint")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sample", type=int, default=0, help="Limiter à N images (0 = tout)")
    ap.add_argument("--oof_from", default=None,
                    help="Dossier des checkpoints de fold: calibre hors-fold "
                         "(seule option correcte après --final_fit)")
    ap.add_argument("--group_column", default=None,
                    help="Colonne de groupe du découpage d'origine (ex: photo_id)")
    ap.add_argument("--k", type=int, default=4, help="Nombre de folds d'origine")
    ap.add_argument("--seed", type=int, default=42, help="Graine du découpage d'origine")
    ap.add_argument("--drop_optional", action="store_true",
                    help="Doit refléter l'entraînement d'origine")
    ap.add_argument("--write", action="store_true",
                    help="Écrit temperature et thresholds DANS le checkpoint")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    state, meta = load_checkpoint(args.checkpoint)
    task = get_task(args.task or meta.task)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.oof_from:
        LOGGER.info("Calibration HORS-FOLD de %s (folds: %s)", args.checkpoint, args.oof_from)
        logits, targets, n_folds = collect_oof_logits(args, task, meta, device)
        n_images = len(targets)
        LOGGER.info("  %d images hors-fold sur %d folds", n_images, n_folds)
    else:
        model = build_backbone(meta.arch, len(meta.classes), pretrained=False)
        model.load_state_dict(state)
        model.to(device).eval()

        df = pd.read_csv(args.csv_path)
        if args.sample:
            df = df.sample(n=min(args.sample, len(df)), random_state=42).reset_index(drop=True)
        label2idx = {c: i for i, c in enumerate(meta.classes)}
        dataset = ClassificationDataset(
            df, args.images_dir, label2idx, task,
            build_eval_transform(meta.img_size, meta.resize_mode, meta.normalize), train=False,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        LOGGER.info("Calibration de %s sur %d images", args.checkpoint, len(df))
        logits, targets = collect_logits(model, loader, device)
        n_images = len(df)

    temperature = fit_temperature(logits, targets, task.multilabel)
    LOGGER.info("Optimal temperature : %.4f  (>1 = the model was overconfident)", temperature)

    if task.multilabel:
        before = torch.sigmoid(logits).numpy()
        after = torch.sigmoid(logits / temperature).numpy()
        y = targets.numpy().astype(int)
        # En multi-label, la confiance d'une prédiction « absente » vaut 1-p:
        # une face annoncée à 0.02 est affirmée absente avec 98 % de confiance.
        def _conf_and_hits(probs):
            predicted = probs >= 0.5
            confidence = np.where(predicted, probs, 1.0 - probs)
            return confidence.ravel(), (predicted == (y == 1)).ravel()

        ece_before = expected_calibration_error(*_conf_and_hits(before))
        ece_after = expected_calibration_error(*_conf_and_hits(after))
        thresholds = best_thresholds(after, y, list(meta.classes))
    else:
        before = torch.softmax(logits, dim=1).numpy()
        after = torch.softmax(logits / temperature, dim=1).numpy()
        y = targets.numpy()
        ece_before = expected_calibration_error(before.max(1), (before.argmax(1) == y))
        ece_after = expected_calibration_error(after.max(1), (after.argmax(1) == y))
        thresholds = {}

    LOGGER.info("ECE before : %.4f", ece_before)
    LOGGER.info("ECE after  : %.4f  (%+.1f %%)", ece_after,
                100 * (ece_after - ece_before) / max(ece_before, 1e-9))
    for name, thr in thresholds.items():
        LOGGER.info("  seuil %-8s %.2f", name, thr)

    report = {
        "checkpoint": args.checkpoint, "task": task.name, "n_images": n_images,
        "method": "out-of-fold" if args.oof_from else "holdout",
        "temperature": temperature, "thresholds": thresholds,
        "ece_before": ece_before, "ece_after": ece_after,
    }
    out = Path(args.checkpoint).with_suffix(".calibration.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Rapport -> %s", out)

    if args.write:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        ckpt["temperature"] = temperature
        ckpt["thresholds"] = thresholds
        torch.save(ckpt, args.checkpoint)
        LOGGER.info("Checkpoint updated: temperature + thresholds written in.")


if __name__ == "__main__":
    main()
