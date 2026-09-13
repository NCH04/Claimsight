"""Format de checkpoint unifié + chargement sûr.

Le checkpoint transporte son propre contrat de préprocessing (`img_size`,
`resize_mode`, `normalize`). L'inférence reconstruit le transform à partir de
ces métadonnées: il devient impossible de servir un modèle avec un
préprocessing différent de celui de son entraînement.

Sécurité: `torch.load(..., weights_only=True)` interdit la désérialisation
d'objets pickle arbitraires (exécution de code à la simple lecture d'un .pt).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

CHECKPOINT_FORMAT_VERSION = 2


@dataclass
class CheckpointMeta:
    """Métadonnées décrivant comment reconstruire et servir un modèle."""

    arch: str
    classes: list[str]
    img_size: int = 256
    resize_mode: str = "center_crop"
    normalize: str = "imagenet"
    task: str = "unknown"
    #: Multi-label (sigmoïde + seuil) plutôt que mono-label (softmax + argmax).
    #: Sans cette information l'inférence servirait un modèle BCE comme un
    #: modèle CrossEntropy et ne renverrait qu'une face sur deux.
    multilabel: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    format_version: int = CHECKPOINT_FORMAT_VERSION


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    meta: CheckpointMeta,
    extra: dict[str, Any] | None = None,
) -> None:
    """Sauvegarde un checkpoint auto-descriptif.

    Les clés legacy (`model_state`, `labels`) sont conservées pour rester
    lisible par l'ancien code.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    state = {k: v.cpu() for k, v in model.state_dict().items()}
    payload: dict[str, Any] = {
        "model": state,
        **asdict(meta),
        # --- rétrocompat descendante ---
        "model_state": state,
        "labels": list(meta.classes),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, out)


def load_checkpoint(path: str | Path) -> tuple[dict[str, Any], CheckpointMeta]:
    """Charge un checkpoint (nouveau ou ancien format) et normalise ses métadonnées.

    Returns:
        (state_dict, meta)

    Raises:
        FileNotFoundError: chemin absent.
        ValueError: contenu inexploitable (poids ou classes manquants).
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - dépend du fichier fourni
        raise ValueError(
            f"Lecture impossible de {ckpt_path} en mode sûr (weights_only=True): {exc}. "
            "Si ce checkpoint vient d'une source de confiance et d'un ancien format, "
            "re-sauvegardez-le avec claimsight.ml.checkpoint.save_checkpoint."
        ) from exc

    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint {ckpt_path}: dict attendu, reçu {type(ckpt).__name__}")

    # `or` serait piégeux ici (un state_dict vide est falsy) -> test d'appartenance.
    state = ckpt["model"] if "model" in ckpt else ckpt.get("model_state")
    classes: Sequence[str] | None = (
        ckpt["classes"] if "classes" in ckpt else ckpt.get("labels")
    )

    if state is None or classes is None:
        raise ValueError(
            f"Checkpoint invalide: {ckpt_path}\n"
            f"  Clés attendues : model, classes, arch, img_size, resize_mode, normalize\n"
            f"  Legacy accepté : model_state, labels\n"
            f"  Clés trouvées  : {sorted(ckpt.keys())}"
        )

    meta = CheckpointMeta(
        arch=str(ckpt.get("arch", "resnet18")),
        classes=[str(c) for c in classes],
        img_size=int(ckpt.get("img_size", 224)),
        # Les checkpoints v1 n'ont pas de resize_mode: ils ont été servis en
        # center_crop, on préserve ce comportement pour ne pas changer leurs
        # prédictions silencieusement.
        resize_mode=str(ckpt.get("resize_mode", "center_crop")),
        normalize=str(ckpt.get("normalize", "imagenet")),
        task=str(ckpt.get("task", "unknown")),
        multilabel=bool(ckpt.get("multilabel", False)),
        metrics=dict(ckpt.get("metrics", {}) or {}),
        format_version=int(ckpt.get("format_version", 1)),
    )
    return state, meta
