import os
import argparse
import random
import pandas as pd
import numpy as np

from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Map pour inverser les labels lorsqu'on applique un flip horizontal volontaire
DIR_MIRROR_MAP = {
    "front-left": "front-right",
    "front-right": "front-left",
    "left": "right",
    "right": "left",
    "rear-left": "rear-right",
    "rear-right": "rear-left",
}


def build_model(backbone: str, num_classes: int):
    """Construit un modèle torchvision avec la bonne tête de classification."""
    backbone = backbone.lower()
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    elif backbone == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError(f"Backbone non supporté: {backbone}")
    return model


class ViewDataset(Dataset):
    def __init__(self, df, images_dir, label2idx, train=True, img_size=224, mirror_prob=0.0):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.label2idx = label2idx
        self.train = train
        self.mirror_prob = mirror_prob

        # Augmentations simples et safe (pas de flip pour éviter left/right swap)
        if self.train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path_ls = row["image"]
        label = row["view"]

        # Label Studio path -> local file
        fname = os.path.basename(img_path_ls)
        img_path = os.path.join(self.images_dir, fname)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image introuvable: {img_path}\n"
                                    f"-> Vérifie images_dir et que les fichiers ont les mêmes noms.")

        img = Image.open(img_path).convert("RGB")

        # Flip horizontal avec swap de label pour mieux apprendre gauche/droite
        if (
            self.train
            and self.mirror_prob > 0
            and label in DIR_MIRROR_MAP
            and random.random() < self.mirror_prob
        ):
            img = ImageOps.mirror(img)
            label = DIR_MIRROR_MAP[label]

        x = self.tf(img)
        y = self.label2idx[label]
        return x, y


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.argmax(logits, dim=1).cpu().numpy()
        preds.extend(p.tolist())
        ys.extend(y.numpy().tolist())
    return np.array(ys), np.array(preds)


def set_backbone_trainable(model, trainable: bool):
    """Active/désactive le fine-tuning du backbone (garde la couche fc entraînable)."""
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    # Assure la reproductibilité dans les workers DataLoader
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def main(args):
    set_seed(args.seed)

    df = pd.read_csv(args.csv_path)

    # On ne garde que image + view
    df = df[["image", "view"]].dropna()
    # Exclut les closeup (non utilisées pour l'entraînement des vues)
    df = df[df["view"] != "closeup"]
    if args.drop_out_of_scope:
        df = df[df["view"] != "out_of_scope"]
    
    print(len(df))

    # Labels
    labels = sorted(df["view"].unique().tolist())
    label2idx = {l: i for i, l in enumerate(labels)}
    idx2label = {i: l for l, i in label2idx.items()}

    print("Classes:", labels)
    print("Nb images:", len(df))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    g = torch.Generator()
    g.manual_seed(args.seed)

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["view"])):
        print(f"\n===== FOLD {fold+1}/{args.k} =====")

        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

        train_ds = ViewDataset(
            df_train, args.images_dir, label2idx,
            train=True, img_size=args.img_size, mirror_prob=args.mirror_prob
        )
        val_ds   = ViewDataset(
            df_val, args.images_dir, label2idx,
            train=False, img_size=args.img_size, mirror_prob=0.0
        )

        # Poids de classes (inverse freq) pour limiter le biais si dataset déséquilibré
        counts = df_train["view"].value_counts()
        class_weights = []
        for l in labels:
            w = len(df_train) / (len(labels) * counts.get(l, 1))
            class_weights.append(w)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # Sampler optionnel pour suréchantillonner les classes rares
        if args.use_weighted_sampler:
            class_weight_map = {l: class_weights[label2idx[l]].item() for l in labels}
            sample_weights = df_train["view"].map(class_weight_map).to_numpy()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
            )
        val_loader   = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # Modèle pré-entraîné
        model = build_model(args.backbone, num_classes=len(labels))
        model = model.to(device)

        if args.freeze_backbone_epochs > 0:
            set_backbone_trainable(model, False)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )

        def make_optimizer():
            return torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

        optimizer = make_optimizer()
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )

        best_f1 = -1.0
        os.makedirs(args.out_dir, exist_ok=True)
        best_path = os.path.join(args.out_dir, f"best_fold_{fold}.pt")

        for epoch in range(1, args.epochs + 1):
            # Dégèle le backbone après warmup éventuel
            if args.freeze_backbone_epochs and epoch == args.freeze_backbone_epochs + 1:
                set_backbone_trainable(model, True)
                optimizer = make_optimizer()
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
                )

            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            y_true, y_pred = eval_model(model, val_loader, device)

            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="macro")

            print(f"Epoch {epoch:02d} | loss={loss:.4f} | acc={acc:.4f} | macroF1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                torch.save({
                    # Nouveau format (compatible pipeline)
                    "model": model.state_dict(),
                    "classes": labels,
                    "arch": args.backbone,
                    "img_size": args.img_size,
                    "normalize": "imagenet",
                    # Ancien format (compatibilité descendante)
                    "model_state": model.state_dict(),
                    "labels": labels,
                    "label2idx": label2idx,
                }, best_path)

            scheduler.step()

        # Rapport final fold
        y_true, y_pred = eval_model(model, val_loader, device)
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro")
        cm  = confusion_matrix(y_true, y_pred)

        print("\nConfusion matrix:\n", cm)
        print("\nClassification report:\n", classification_report(
            y_true, y_pred, target_names=[idx2label[i] for i in range(len(labels))]
        ))

        fold_metrics.append((acc, f1))

    accs = [m[0] for m in fold_metrics]
    f1s  = [m[1] for m in fold_metrics]
    print("\n===== Résumé K-Fold =====")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Macro-F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True, help="Path vers le CSV exporté de Label Studio")
    ap.add_argument("--images_dir", type=str, required=True, help="Dossier local contenant les images")
    ap.add_argument("--out_dir", type=str, default="outputs_view", help="Dossier de sortie")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lr_step_size", type=int, default=8, help="StepLR: nb d'epochs avant de réduire le LR")
    ap.add_argument("--lr_gamma", type=float, default=0.1, help="StepLR: facteur de réduction du LR")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--freeze_backbone_epochs", type=int, default=0, help="Nb d'epochs de warmup avec backbone gelé (0 pour désactiver)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--mirror_prob", type=float, default=0.3, help="Proba d'appliquer un flip horizontal + swap gauche/droite")
    ap.add_argument("--use_weighted_sampler", action="store_true", help="Active un suréchantillonnage des classes rares")
    ap.add_argument("--backbone", type=str, default="resnet34", choices=["resnet18", "resnet34"], help="Backbone torchvision à utiliser")
    ap.add_argument("--seed", type=int, default=42, help="Seed pour la reproductibilité")
    ap.add_argument("--drop_out_of_scope", action="store_true", help="Exclut les vues out_of_scope si présentes")
    args = ap.parse_args()
    main(args)
