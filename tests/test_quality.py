"""Tests des contrôles qualité image — numpy et PIL seulement."""

import numpy as np
import pytest
from PIL import Image, ImageEnhance, ImageFilter

from claimsight.domain.quality import (
    BLURRY,
    DARK,
    LOW_CONFIDENCE,
    OK,
    blur_score,
    measure,
    quality_flag,
)


def sharp_image(size=(320, 240)) -> Image.Image:
    """Damier: beaucoup de transitions franches, donc net par construction.

    Le fond est posé AVANT les lignes claires: un `+= 60` sur un uint8 ferait
    déborder les 255 vers 59 et effacerait le contraste qu'on veut mesurer.
    """
    a = np.full((size[1], size[0], 3), 60, dtype=np.uint8)
    a[::8, :] = 255
    a[:, ::8] = 255
    return Image.fromarray(a)


# ------------------------------------------------------------------- mesures
def test_blur_collapses_when_the_image_is_blurred():
    sharp = sharp_image()
    blurred = sharp.filter(ImageFilter.GaussianBlur(5))
    assert measure(blurred).blur < measure(sharp).blur / 10


def test_blur_score_handles_a_degenerate_image():
    assert blur_score(np.zeros((2, 2), dtype=np.float32)) == 0.0


def test_brightness_follows_exposure():
    sharp = sharp_image()
    dark = ImageEnhance.Brightness(sharp).enhance(0.1)
    assert measure(dark).brightness < measure(sharp).brightness


def test_metrics_are_resolution_independent():
    """Les mesures sont prises sur une version réduite: un agrandissement
    ne doit pas changer le verdict."""
    small = sharp_image((160, 120))
    large = small.resize((1280, 960), Image.NEAREST)
    assert quality_flag(measure(small)) == quality_flag(measure(large))


# ------------------------------------------------------------------- verdict
def test_sharp_and_well_exposed_is_ok():
    assert quality_flag(measure(sharp_image())) == OK


def test_blurred_is_flagged():
    assert quality_flag(measure(sharp_image().filter(ImageFilter.GaussianBlur(6)))) == BLURRY


def test_dark_wins_over_blur():
    """Régression: l'obscurité écrase la variance du Laplacien, donc une image
    sombre tombe AUSSI sous le seuil de flou. Le défaut actionnable étant
    l'exposition, `dark` doit l'emporter quand les deux seuils sont franchis."""
    from claimsight.domain.quality import ImageQuality

    both = ImageQuality(blur=2.0, brightness=12.0)   # sous les deux seuils
    assert quality_flag(both) == DARK


def test_a_real_darkened_photo_is_not_called_blurry():
    """Sur une vraie photo assombrie, le verdict ne doit pas être `blurry`."""
    dark = ImageEnhance.Brightness(sharp_image()).enhance(0.05)
    assert quality_flag(measure(dark)) in (DARK, OK)


def test_low_confidence_only_when_the_image_itself_is_fine():
    good = measure(sharp_image())
    assert quality_flag(good, view_confidence=0.2, min_confidence=0.5) == LOW_CONFIDENCE
    assert quality_flag(good, view_confidence=0.9, min_confidence=0.5) == OK


def test_threshold_of_zero_disables_the_confidence_check():
    good = measure(sharp_image())
    assert quality_flag(good, view_confidence=0.01, min_confidence=0.0) == OK


@pytest.mark.parametrize("blur,bright,expected",
                         [(10.0, 200.0, BLURRY), (500.0, 10.0, DARK), (500.0, 200.0, OK)])
def test_thresholds_are_configurable(blur, bright, expected):
    from claimsight.domain.quality import ImageQuality
    assert quality_flag(ImageQuality(blur=blur, brightness=bright)) == expected
