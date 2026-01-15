import os
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import models, transforms

from .config import DEFAULT_DAMAGE_CLASSES, IMAGENET_MEAN, IMAGENET_STD


def build_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    elif arch == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model


class DamageModel:
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.is_dummy = checkpoint_path is None

        # Dummy by default if no checkpoint provided or not found
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            self.model = None
            self.classes = DEFAULT_DAMAGE_CLASSES
            self.tf = None
            self.is_dummy = True
            return

        self.is_dummy = False
        self.model, self.classes, self.tf = self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self) -> Tuple[nn.Module, List[str], transforms.Compose]:
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        keys = ckpt.keys()
        state_dict = ckpt.get("model") or ckpt.get("model_state")
        classes: List[str] = ckpt.get("classes") or ckpt.get("labels") or DEFAULT_DAMAGE_CLASSES
        arch = ckpt.get("arch", "resnet18")
        img_size = int(ckpt.get("img_size", 224))
        normalize = ckpt.get("normalize", "imagenet")

        if state_dict is None:
            raise ValueError(
                f"Invalid damage checkpoint at {self.checkpoint_path}. "
                f"Expected keys: model/classes/arch/img_size/normalize (fallback model_state/labels). "
                f"Found keys: {sorted(list(keys))}"
            )

        model = build_model(arch, num_classes=len(classes))
        model.load_state_dict(state_dict)

        tf_list = [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
        if normalize == "imagenet":
            tf_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        tf = transforms.Compose(tf_list)
        return model, classes, tf

    def predict(self, image) -> Tuple[str, float]:
        if self.is_dummy or self.model is None or self.tf is None:
            return "unknown", 0.0
        x = self.tf(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
        return self.classes[idx.item()], conf.item()
