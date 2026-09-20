"""Construction des backbones torchvision.

Remplace les 4 copies identiques de `build_model()` qui traînaient dans
train_view_kfold.py, view_model.py, damage_model.py et severity_model.py.

`pretrained` est explicite et par défaut à False: à l'inférence on charge de
toute façon un state_dict par-dessus, télécharger ~87 Mo de poids ImageNet pour
les écraser aussitôt coûte du réseau et des secondes au démarrage.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

#: nom -> (constructeur, enum de poids ImageNet)
_REGISTRY: dict[str, tuple] = {
    "resnet18": (models.resnet18, models.ResNet18_Weights),
    "resnet34": (models.resnet34, models.ResNet34_Weights),
    "resnet50": (models.resnet50, models.ResNet50_Weights),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights),
}

SUPPORTED_BACKBONES = tuple(_REGISTRY)


def _replace_head(model: nn.Module, arch: str, num_classes: int) -> nn.Module:
    """Remplace la couche de classification par une tête à `num_classes`."""
    if arch.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch.startswith("efficientnet"):
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, num_classes)
    elif arch.startswith("convnext"):
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, num_classes)
    else:  # pragma: no cover - garde-fou si _REGISTRY grandit sans mise à jour
        raise ValueError(f"Unknown classification head for architecture: {arch}")
    return model


def build_backbone(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Instancie `arch` avec une tête à `num_classes` sorties.

    Args:
        arch: une des clés de SUPPORTED_BACKBONES.
        num_classes: taille de la couche de sortie.
        pretrained: True à l'entraînement (transfer learning), False à
            l'inférence (les poids viennent du checkpoint).
    """
    key = arch.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Backbone non supporté: {arch!r}. Disponibles: {', '.join(SUPPORTED_BACKBONES)}"
        )
    ctor, weights_enum = _REGISTRY[key]
    model = ctor(weights=weights_enum.DEFAULT if pretrained else None)
    return _replace_head(model, key, num_classes)


def backbone_head_prefixes(arch: str) -> tuple:
    """Préfixes des paramètres appartenant à la tête (pour le freeze du backbone)."""
    key = arch.lower()
    if key.startswith("resnet"):
        return ("fc.",)
    return ("classifier.",)
