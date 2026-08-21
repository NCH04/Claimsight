"""Entraînement K-fold générique pour view / damage / severity.

    python -m claimsight.training.train_classifier --task view \
        --csv_path dataset/labels.csv --images_dir dataset/images_mapped

Corrections méthodologiques par rapport à l'ancien entraîneur mono-tâche:

1. Split groupé par véhicule (`--group_column vehicle_id`). Sans groupes, deux
   photos du même sinistre tombent de part et d'autre du split et le modèle
   « reconnaît la voiture » au lieu d'apprendre la vue -> métriques gonflées.
2. Sélection du meilleur epoch sur un holdout découpé DANS le train
   (`--holdout_ratio`), pas sur le fold de validation. Sélectionner et
   rapporter sur le même ensemble biaise le score vers le haut.
3. Le rapport final est celui du MEILLEUR checkpoint. L'ancien code
   réévaluait le modèle après la boucle, donc l'état du DERNIER epoch: les
   métriques publiées ne décrivaient pas le .pt sauvegardé.
4. Le transform de validation est celui de l'inférence (claimsight.ml.transforms).
5. `random_state` suit `--seed` (il était figé à 42).
6. Métriques persistées en JSON, et `--final_fit` réentraîne sur 100% des
   données pour produire le modèle déployable.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..ml.backbones import SUPPORTED_BACKBONES, backbone_head_prefixes, build_backbone
from ..ml.checkpoint import CheckpointMeta, save_checkpoint
from ..ml.transforms import RESIZE_MODES, build_eval_transform, build_train_transform
from .dataset import ClassificationDataset
from .metrics import aggregate_folds, compute_metrics, format_confusion, full_report
from .tasks import TASKS, TaskConfig, get_task

LOGGER = logging.getLogger("train")


# --------------------------------------------------------------------------- setup
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool) -> None:
    head = backbone_head_prefixes(arch)
    for name, param in model.named_parameters():
        if not name.startswith(head):
            param.requires_grad = trainable


# --------------------------------------------------------------------------- data
def load_dataframe(args, task: TaskConfig) -> tuple[pd.DataFrame, list[str], str | None]:
    df = pd.read_csv(args.csv_path)

    required = {"image", task.label_column}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Colonnes manquantes dans {args.csv_path}: {sorted(missing)}. "
            f"Colonnes présentes: {sorted(df.columns)}"
        )

    cols = ["image", task.label_column]
    group_col = args.group_column if args.group_column in df.columns else None
    if args.group_column and group_col is None:
        LOGGER.warning(
            "Colonne de groupe %r absente du CSV -> split NON groupé. "
            "Si plusieurs photos partagent un véhicule, les métriques seront optimistes.",
            args.group_column,
        )
    if group_col:
        cols.append(group_col)

    df = df[cols].dropna(subset=["image", task.label_column])
    df[task.label_column] = df[task.label_column].astype(str).str.strip()

    before = len(df)
    drop = set(task.drop_labels)
    if args.drop_optional:
        drop |= set(task.optional_drop_labels)
    if drop:
        df = df[~df[task.label_column].isin(drop)]
        LOGGER.info("Labels exclus %s -> %d lignes retirées", sorted(drop), before - len(df))

    unknown = set(df[task.label_column].unique()) - set(task.classes)
    if unknown:
        raise SystemExit(
            f"Labels hors taxonomie pour la tâche {task.name!r}: {sorted(unknown)}\n"
            f"Attendus: {list(task.classes)}\n"
            f"Corrigez le CSV ou claimsight/domain/taxonomy.py."
        )

    # On n'entraîne que sur les classes réellement présentes: une classe à 0
    # exemple produirait une colonne de logits jamais supervisée.
    present = [c for c in task.classes if c in set(df[task.label_column])]
    counts = df[task.label_column].value_counts()
    too_rare = [c for c in present if counts[c] < args.k]
    if too_rare:
        raise SystemExit(
            f"Classes avec moins de {args.k} exemples (impossible à stratifier sur {args.k} folds): "
            f"{ {c: int(counts[c]) for c in too_rare} }\n"
            f"Réduisez --k, fusionnez ces classes, ou collectez plus de données."
        )
    return df.reset_index(drop=True), present, group_col


def _loader_kwargs(args) -> dict:
    return dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )


def make_train_loader(df_train, args, task, label2idx, train_tf) -> DataLoader:
    ds = ClassificationDataset(
        df_train, args.images_dir, label2idx, task, train_tf,
        train=True, mirror_prob=args.mirror_prob,
    )
    g = torch.Generator()
    g.manual_seed(args.seed)

    if args.use_weighted_sampler:
        counts = df_train[task.label_column].value_counts()
        weights = df_train[task.label_column].map(lambda lbl: 1.0 / counts[lbl]).to_numpy()
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=g,
        )
        return DataLoader(ds, sampler=sampler, generator=g, **_loader_kwargs(args))
    return DataLoader(ds, shuffle=True, generator=g, **_loader_kwargs(args))


def make_eval_loader(df_eval, args, task, label2idx, eval_tf) -> DataLoader:
    ds = ClassificationDataset(
        df_eval, args.images_dir, label2idx, task, eval_tf, train=False,
    )
    return DataLoader(ds, shuffle=False, **_loader_kwargs(args))


def class_weights_for(df_train, labels, task, device) -> torch.Tensor:
    counts = df_train[task.label_column].value_counts()
    n, k = len(df_train), len(labels)
    return torch.tensor(
        [n / (k * max(int(counts.get(lbl, 0)), 1)) for lbl in labels],
        dtype=torch.float32,
        device=device,
    )


# --------------------------------------------------------------------------- loops
def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.inference_mode()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        ys.extend(y.tolist())
    return np.asarray(ys), np.asarray(preds)


def fit(
    model, train_loader, sel_loader, criterion, args, labels, task, device,
) -> tuple[nn.Module, dict]:
    """Entraîne et retourne (modèle au meilleur epoch, historique)."""
    arch = args.backbone

    def make_optimizer():
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    if args.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, arch, False)

    optimizer = make_optimizer()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )

    best_score, best_epoch, best_state, since_improved = -1.0, 0, None, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbone_epochs and epoch == args.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, arch, True)
            optimizer = make_optimizer()
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
            )
            LOGGER.info("  backbone dégelé (epoch %d)", epoch)

        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        if sel_loader is None:
            history.append({"epoch": epoch, "loss": loss})
            LOGGER.info("  epoch %02d | loss=%.4f", epoch, loss)
            best_epoch = epoch
            continue

        y_true, y_pred = evaluate(model, sel_loader, device)
        m = compute_metrics(y_true, y_pred, labels, ordinal=task.ordinal)
        score = m["macro_f1"]
        history.append({"epoch": epoch, "loss": loss, **m})
        LOGGER.info(
            "  epoch %02d | loss=%.4f | sel_acc=%.4f | sel_macroF1=%.4f",
            epoch, loss, m["accuracy"], score,
        )

        if score > best_score:
            best_score, best_epoch, since_improved = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_improved += 1
            if args.early_stopping_patience and since_improved >= args.early_stopping_patience:
                LOGGER.info("  early stopping (aucun gain depuis %d epochs)", since_improved)
                break

    # On restaure le MEILLEUR état: c'est lui qu'on évalue et qu'on sauvegarde.
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_epoch": best_epoch,
        "best_selection_macro_f1": best_score if best_score >= 0 else None,
        "epochs_ran": len(history),
        "history": history,
    }


# --------------------------------------------------------------------------- main
def build_splitter(args, group_col):
    if group_col:
        return StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=args.seed), True
    return StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed), False


def run(args) -> dict:
    task = get_task(args.task)
    set_seed(args.seed)

    df, labels, group_col = load_dataframe(args, task)
    label2idx = {lbl: i for i, lbl in enumerate(labels)}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    LOGGER.info("Tâche      : %s — %s", task.name, task.description)
    LOGGER.info("Images     : %d | classes: %d %s", len(df), len(labels), labels)
    LOGGER.info("Répartition: %s", df[task.label_column].value_counts().to_dict())
    LOGGER.info("Device     : %s | backbone: %s | img_size: %d | resize: %s",
                device, args.backbone, args.img_size, args.resize_mode)
    LOGGER.info("Split      : %s", f"groupé par {group_col!r}" if group_col else "PAR IMAGE (non groupé)")

    train_tf = build_train_transform(args.img_size, args.resize_mode)
    eval_tf = build_eval_transform(args.img_size, args.resize_mode)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splitter, grouped = build_splitter(args, group_col)
    groups = df[group_col] if grouped else None

    fold_metrics: list[dict] = []
    fold_details: list[dict] = []
    best_epochs: list[int] = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(df, df[task.label_column], groups)):
        LOGGER.info("\n===== FOLD %d/%d =====", fold + 1, args.k)
        df_tr_full = df.iloc[tr_idx].copy()
        df_va = df.iloc[va_idx].copy()

        # Holdout de sélection découpé DANS le train: garde le fold de
        # validation vierge de toute décision de modèle.
        sel_df = None
        if args.holdout_ratio > 0:
            try:
                df_tr, sel_df = train_test_split(
                    df_tr_full,
                    test_size=args.holdout_ratio,
                    stratify=df_tr_full[task.label_column],
                    random_state=args.seed,
                )
            except ValueError:
                LOGGER.warning(
                    "  holdout stratifié impossible (classe trop rare) -> "
                    "sélection sur le fold de validation (score optimiste)."
                )
                df_tr, sel_df = df_tr_full, df_va
        else:
            df_tr, sel_df = df_tr_full, df_va

        train_loader = make_train_loader(df_tr, args, task, label2idx, train_tf)
        sel_loader = make_eval_loader(sel_df, args, task, label2idx, eval_tf)
        val_loader = make_eval_loader(df_va, args, task, label2idx, eval_tf)

        model = build_backbone(args.backbone, len(labels), pretrained=not args.no_pretrained)
        model.to(device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_for(df_tr, labels, task, device) if args.class_weights else None,
            label_smoothing=args.label_smoothing,
        )

        model, hist = fit(model, train_loader, sel_loader, criterion, args, labels, task, device)
        best_epochs.append(hist["best_epoch"])

        y_true, y_pred = evaluate(model, val_loader, device)
        m = compute_metrics(y_true, y_pred, labels, ordinal=task.ordinal)
        rep = full_report(y_true, y_pred, labels)
        fold_metrics.append(m)
        fold_details.append({"fold": fold, **m, **hist, **rep})

        LOGGER.info("  [fold %d] VALIDATION acc=%.4f macroF1=%.4f (best_epoch=%d)",
                    fold + 1, m["accuracy"], m["macro_f1"], hist["best_epoch"])
        LOGGER.info("\n%s", format_confusion(rep["confusion_matrix"], labels))

        ckpt_path = out_dir / f"{task.name}_fold{fold}.pt"
        save_checkpoint(
            ckpt_path,
            model,
            CheckpointMeta(
                arch=args.backbone, classes=labels, img_size=args.img_size,
                resize_mode=args.resize_mode, normalize="imagenet",
                task=task.name, metrics=m,
            ),
        )

    summary = aggregate_folds(fold_metrics)
    LOGGER.info("\n===== RÉSUMÉ K-FOLD (%s) =====", task.name)
    for k, v in summary.items():
        LOGGER.info("  %-22s %.4f ± %.4f", k, v["mean"], v["std"])

    results = {
        "task": task.name,
        "classes": labels,
        "n_images": len(df),
        "grouped_split": bool(grouped),
        "group_column": group_col,
        "config": {
            k: v for k, v in vars(args).items() if k not in ("func",)
        },
        "cv_summary": summary,
        "folds": fold_details,
    }

    # Modèle déployable: réentraîné sur 100% des données, au nombre d'epochs
    # médian retenu par la CV. La CV donne l'estimation, ce fit donne le modèle.
    if args.final_fit:
        n_epochs = int(np.median(best_epochs)) or 1
        LOGGER.info("\n===== FINAL FIT (toutes les données, %d epochs) =====", n_epochs)
        final_args = copy.copy(args)
        final_args.epochs = n_epochs
        final_args.early_stopping_patience = 0
        train_loader = make_train_loader(df, args, task, label2idx, train_tf)
        model = build_backbone(args.backbone, len(labels), pretrained=not args.no_pretrained).to(device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_for(df, labels, task, device) if args.class_weights else None,
            label_smoothing=args.label_smoothing,
        )
        model, _ = fit(model, train_loader, None, criterion, final_args, labels, task, device)
        final_path = out_dir / f"{task.name}.pt"
        save_checkpoint(
            final_path, model,
            CheckpointMeta(
                arch=args.backbone, classes=labels, img_size=args.img_size,
                resize_mode=args.resize_mode, normalize="imagenet", task=task.name,
                metrics={k: v["mean"] for k, v in summary.items()},
            ),
        )
        results["final_model"] = str(final_path)
        results["final_epochs"] = n_epochs
        LOGGER.info("Modèle déployable -> %s", final_path)

    metrics_path = out_dir / f"{task.name}_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Métriques -> %s", metrics_path)
    return results


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Entraînement K-fold générique (view / damage / severity).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--task", required=True, choices=sorted(TASKS), help="Tâche à entraîner")
    ap.add_argument("--csv_path", required=True, help="CSV: colonnes image, <label>, [vehicle_id]")
    ap.add_argument("--images_dir", required=True, help="Dossier des images")
    ap.add_argument("--out_dir", default=None, help="Défaut: outputs/train/<task>")

    ap.add_argument("--k", type=int, default=5, help="Nombre de folds")
    ap.add_argument("--group_column", default="vehicle_id",
                    help="Colonne de groupe pour éviter la fuite entre folds")
    ap.add_argument("--holdout_ratio", type=float, default=0.15,
                    help="Part du train réservée à la sélection d'epoch (0 = sélection sur la val, optimiste)")

    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lr_step_size", type=int, default=8)
    ap.add_argument("--lr_gamma", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--early_stopping_patience", type=int, default=8, help="0 = désactivé")
    ap.add_argument("--freeze_backbone_epochs", type=int, default=1, help="0 = désactivé")

    ap.add_argument("--backbone", default=None, choices=list(SUPPORTED_BACKBONES))
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--resize_mode", default=None, choices=list(RESIZE_MODES))
    ap.add_argument("--no_pretrained", action="store_true", help="Désactive l'init ImageNet")

    ap.add_argument("--mirror_prob", type=float, default=0.5,
                    help="Proba de flip horizontal (avec permutation du label si la tâche l'exige)")
    ap.add_argument("--use_weighted_sampler", action="store_true")
    ap.add_argument("--class_weights", action="store_true", default=True)
    ap.add_argument("--no_class_weights", dest="class_weights", action="store_false")
    ap.add_argument("--drop_optional", action="store_true",
                    help="Exclut aussi les labels optionnels de la tâche (ex: out_of_scope)")

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--final_fit", action="store_true",
                    help="Réentraîne sur 100%% des données -> <out_dir>/<task>.pt déployable")
    ap.add_argument("--log_level", default="INFO")
    return ap


def apply_task_defaults(args) -> None:
    """Les valeurs par défaut dépendent de la tâche (cf. claimsight/training/tasks.py)."""
    task = get_task(args.task)
    if args.backbone is None:
        args.backbone = task.default_backbone
    if args.img_size is None:
        args.img_size = task.default_img_size
    if args.resize_mode is None:
        args.resize_mode = task.default_resize_mode
    if args.out_dir is None:
        args.out_dir = f"outputs/train/{task.name}"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    apply_task_defaults(args)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
