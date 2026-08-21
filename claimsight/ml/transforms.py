"""Source unique des transforms image.

Bug corrigé ici: le pipeline utilisait trois géométries différentes pour un
même modèle.
  - train  : RandomResizedCrop(s)                      -> aspect ~conservé
  - val    : Resize((s, s))                            -> aspect ÉCRASÉ
  - infer  : Resize(s) + CenterCrop(s)                 -> aspect conservé
Le modèle était donc validé sur des images écrasées puis servi sur des images
recadrées. `build_eval_transform` est désormais l'unique chemin pour la
validation ET l'inférence, et le mode de redimensionnement est stocké dans le
checkpoint (voir ml/checkpoint.py) pour qu'ils ne puissent plus diverger.
"""

from __future__ import annotations

from PIL import Image, ImageOps
from torchvision import transforms

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: center_crop : redimensionne le petit côté puis recadre au centre. Standard
#:               ImageNet, mais rogne les bords d'une photo 4:3 (~25% de la
#:               largeur) — peut couper l'avant ou l'arrière du véhicule.
#: pad_square  : letterbox, conserve 100% du cadre en ajoutant des bandes.
#:               Recommandé pour la classification de vue, où le véhicule
#:               entier porte l'information.
RESIZE_MODES = ("center_crop", "pad_square")


class PadToSquare:
    """Letterbox: complète l'image en carré sans rien rogner."""

    def __init__(self, fill: int = 0) -> None:
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        left = (side - w) // 2
        top = (side - h) // 2
        # (left, top, right, bottom)
        padding = (left, top, side - w - left, side - h - top)
        return ImageOps.expand(img, border=padding, fill=self.fill)


def _geometry(img_size: int, resize_mode: str):
    if resize_mode == "center_crop":
        return [transforms.Resize(img_size), transforms.CenterCrop(img_size)]
    if resize_mode == "pad_square":
        return [PadToSquare(), transforms.Resize((img_size, img_size))]
    raise ValueError(f"resize_mode inconnu: {resize_mode!r}. Attendu: {RESIZE_MODES}")


def _normalize(normalize: str):
    if normalize == "imagenet":
        return [transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if normalize in ("none", None):
        return []
    raise ValueError(f"normalize inconnu: {normalize!r}")


def build_eval_transform(
    img_size: int = 256,
    resize_mode: str = "center_crop",
    normalize: str = "imagenet",
) -> transforms.Compose:
    """Transform déterministe, utilisé pour la validation ET l'inférence."""
    steps = _geometry(img_size, resize_mode)
    steps.append(transforms.ToTensor())
    steps.extend(_normalize(normalize))
    return transforms.Compose(steps)


def build_train_transform(
    img_size: int = 256,
    resize_mode: str = "center_crop",
    normalize: str = "imagenet",
    scale: tuple[float, float] = (0.9, 1.0),
    color_jitter: float = 0.15,
    rotation_degrees: float = 5.0,
    translate: float = 0.02,
) -> transforms.Compose:
    """Transform d'entraînement.

    Le flip horizontal n'est PAS inclus: il est géré au niveau du dataset, car
    pour la tâche `view` il faut aussi permuter le label (left <-> right).
    """
    if resize_mode == "pad_square":
        pre = [PadToSquare(), transforms.RandomResizedCrop(img_size, scale=scale)]
    else:
        pre = [transforms.RandomResizedCrop(img_size, scale=scale)]

    steps = pre + [
        transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter),
        transforms.RandomRotation(degrees=rotation_degrees),
        transforms.RandomAffine(degrees=0, translate=(translate, translate)),
        transforms.ToTensor(),
    ]
    steps.extend(_normalize(normalize))
    return transforms.Compose(steps)
