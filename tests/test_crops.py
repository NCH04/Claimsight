"""Tests de la dérivation d'exemples au niveau crop.

Le crop est le domaine dans lequel le classifieur de dégât est *servi*: la
géométrie doit correspondre exactement à celle de `Detection.crop`, sinon on
réintroduit le décalage train/serve que ce chemin existe pour corriger.
"""

import importlib.util
from pathlib import Path

from PIL import Image

from claimsight.ml.detector import Detection

_spec = importlib.util.spec_from_file_location(
    "derive_damage_crops",
    Path(__file__).resolve().parents[1] / "scripts" / "derive_damage_crops.py",
)
derive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(derive)


def test_polygon_bbox_englobe_tous_les_points():
    # Triangle normalisé sur une image 200x100.
    poly = "0.1 0.2 0.5 0.8 0.3 0.4".split()
    assert derive.polygon_bbox(poly, 200, 100) == (20.0, 20.0, 100.0, 80.0)


def test_polygon_bbox_refuse_un_polygone_vide():
    assert derive.polygon_bbox([], 200, 100) is None


def test_photo_id_regroupe_les_augmentations_roboflow():
    """Deux copies augmentées d'une photo doivent partager le même groupe."""
    a = derive.photo_id("000034_jpg.rf.3b4cf87a53b2cd42423b80903ed51aa3.jpg")
    b = derive.photo_id("000034_jpg.rf.9999999999999999999999999999999.jpg")
    assert a == b == "000034_jpg"


def test_la_marge_de_crop_suit_celle_du_service():
    """La constante du script et le défaut de Detection.crop ne doivent pas diverger."""
    image = Image.new("RGB", (100, 100), "white")
    bbox = (40.0, 40.0, 60.0, 60.0)
    det = Detection(label="x", confidence=1.0, bbox=bbox)
    assert det.crop(image, derive.CROP_MARGIN).size == det.crop(image).size


def test_le_crop_garde_du_contexte_autour_de_la_boite():
    """8 % de marge de chaque côté: une boîte de 20 px en rend 23."""
    image = Image.new("RGB", (100, 100), "white")
    det = Detection(label="x", confidence=1.0, bbox=(40.0, 40.0, 60.0, 60.0))
    crop = det.crop(image, derive.CROP_MARGIN)
    assert crop.size == (23, 23)


def test_le_crop_est_borne_par_l_image():
    """Une boîte au bord ne doit pas produire de coordonnées négatives."""
    image = Image.new("RGB", (50, 50), "white")
    det = Detection(label="x", confidence=1.0, bbox=(0.0, 0.0, 50.0, 50.0))
    crop = det.crop(image, derive.CROP_MARGIN)
    assert crop.size == (50, 50)
