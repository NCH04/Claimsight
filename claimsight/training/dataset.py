"""Dataset de classification piloté par une TaskConfig."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .tasks import TaskConfig


def load_rgb(path: str | Path) -> Image.Image:
    """Ouvre une image en RGB en appliquant l'orientation EXIF.

    Sans `exif_transpose`, une photo prise en portrait avec un smartphone est
    servie couchée à 90°: le modèle voit une voiture sur le flanc. C'est le
    cas nominal dans une app d'assurance où tout l'input vient d'un téléphone.
    """
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


class ClassificationDataset(Dataset):
    def __init__(
        self,
        df,
        images_dir: str | Path,
        label2idx: dict[str, int],
        task: TaskConfig,
        transform,
        train: bool = True,
        mirror_prob: float = 0.0,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.label2idx = label2idx
        self.task = task
        self.transform = transform
        self.train = train
        self.mirror_prob = mirror_prob if train else 0.0

    def __len__(self) -> int:
        return len(self.df)

    def resolve_path(self, raw: str) -> Path:
        """Résout une entrée CSV (souvent un chemin Label Studio) en fichier local."""
        candidate = Path(raw)
        if candidate.is_file():
            return candidate
        local = self.images_dir / candidate.name
        if not local.exists():
            raise FileNotFoundError(
                f"Image introuvable: {local}\n"
                f"  (entrée CSV: {raw!r})\n"
                f"  Vérifiez --images_dir et que les noms de fichiers correspondent."
            )
        return local

    def __getitem__(self, i):
        row = self.df.iloc[i]
        label = str(row[self.task.label_column])
        img = load_rgb(self.resolve_path(str(row["image"])))

        if (
            self.mirror_prob > 0
            and self.task.can_mirror(label)
            and random.random() < self.mirror_prob
        ):
            img = ImageOps.mirror(img)
            label = self.task.mirrored_label(label)

        return self.transform(img), self.label2idx[label]
