"""Contrôles qualité d'une photo de sinistre.

Le schéma prévoit un `quality_flag` depuis l'origine, mais il ne valait que
`ok` ou `low_confidence`. Or les deux défauts les plus fréquents sur des photos
prises à la volée en bord de route sont le FLOU et la SOUS-EXPOSITION — et ils
méritent une reprise, pas une prédiction hasardeuse que le gestionnaire devra
démêler ensuite.

numpy et PIL seulement: la couche domaine reste importable sans torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

#: Variance du Laplacien en dessous de laquelle l'image est jugée floue.
#: Calibré sur des photos de téléphone: en dessous de ~60 les arêtes d'un
#: véhicule ne sont plus nettes.
BLUR_THRESHOLD = 60.0

#: Luminance moyenne (0-255) en dessous de laquelle l'image est sous-exposée.
DARK_THRESHOLD = 40.0

#: Les métriques sont calculées sur une version réduite: insensible à la
#: résolution d'origine, et beaucoup plus rapide sur une photo de 12 Mpx.
_ANALYSIS_MAX_SIDE = 512

OK, BLURRY, DARK, LOW_CONFIDENCE = "ok", "blurry", "dark", "low_confidence"


@dataclass(frozen=True)
class ImageQuality:
    blur: float
    brightness: float

    def as_dict(self, ndigits: int = 1) -> dict:
        return {"blur": round(self.blur, ndigits), "brightness": round(self.brightness, ndigits)}


def _to_gray(image: Image.Image, max_side: int = _ANALYSIS_MAX_SIDE) -> np.ndarray:
    gray = image.convert("L")
    if max(gray.size) > max_side:
        scale = max_side / max(gray.size)
        gray = gray.resize(
            (max(1, int(gray.width * scale)), max(1, int(gray.height * scale))),
            Image.BILINEAR,
        )
    return np.asarray(gray, dtype=np.float32)


def blur_score(gray: np.ndarray) -> float:
    """Variance du Laplacien (noyau 4-connexe). Faible = image floue.

    Une image nette contient des transitions brutales, donc un Laplacien à
    forte variance; le flou les lisse et effondre cette variance.
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(laplacian.var())


def measure(image: Image.Image) -> ImageQuality:
    """Métriques qualité d'une image, en une seule conversion."""
    gray = _to_gray(image)
    return ImageQuality(
        blur=blur_score(gray),
        brightness=float(gray.mean()) if gray.size else 0.0,
    )


def quality_flag(
    quality: ImageQuality,
    view_confidence: float | None = None,
    blur_threshold: float = BLUR_THRESHOLD,
    dark_threshold: float = DARK_THRESHOLD,
    min_confidence: float = 0.0,
) -> str:
    """Défaut le plus bloquant de l'image.

    La sous-exposition est testée AVANT le flou, contre l'intuition: une image
    sombre a mécaniquement une faible variance de Laplacien — l'obscurité écrase
    les transitions — et serait donc signalée `blurry` à tort. Or le défaut
    actionnable est bien l'exposition: c'est « reprenez la photo avec plus de
    lumière » qu'il faut dire au gestionnaire, et la mesure de flou n'est de
    toute façon pas fiable sur une image noire.

    Une prédiction peu sûre sur une image par ailleurs correcte vient en dernier.
    """
    if quality.brightness < dark_threshold:
        return DARK
    if quality.blur < blur_threshold:
        return BLURRY
    if view_confidence is not None and min_confidence > 0 and view_confidence < min_confidence:
        return LOW_CONFIDENCE
    return OK
