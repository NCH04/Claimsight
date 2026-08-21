"""Déduplication perceptuelle des photos d'un même dossier.

Le schéma V1 prévoit un champ `deduplicated` depuis le début, mais il était
figé à `False`. Or un dossier de sinistre contient très souvent plusieurs
prises quasi identiques du même angle: sans déduplication, elles pèsent
plusieurs fois dans l'agrégation et gonflent artificiellement la couverture
photo.

Implémentation: dHash (difference hash) 64 bits, en PIL pur — pas de
dépendance supplémentaire. Robuste au redimensionnement, à la compression
JPEG et aux petites variations d'exposition; sensible au recadrage, ce qui
est le comportement voulu (un zoom sur le dégât n'est pas un doublon).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

#: Distance de Hamming maximale (sur 64 bits) pour considérer deux images
#: comme des doublons. 0 = strictement identiques. 5 tolère recompression et
#: légères retouches sans confondre deux angles distincts.
DEFAULT_HAMMING_THRESHOLD = 5

_HASH_SIZE = 8


def dhash(image, hash_size: int = _HASH_SIZE) -> int:
    """Hash perceptuel 64 bits basé sur les gradients horizontaux."""
    # (hash_size + 1) colonnes -> hash_size comparaisons par ligne.
    small = image.convert("L").resize((hash_size + 1, hash_size))
    # tobytes() plutôt que getdata(): en mode "L" il rend exactement
    # width*height octets, et getdata() disparaît dans Pillow 14.
    pixels = small.tobytes()

    bits = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            bits <<= 1
            if pixels[offset + col] > pixels[offset + col + 1]:
                bits |= 1
    return bits


def hamming(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


def find_duplicates(
    hashes: Sequence[int],
    threshold: int = DEFAULT_HAMMING_THRESHOLD,
) -> list[int | None]:
    """Pour chaque image, l'index de l'original dont elle est un doublon.

    La PREMIÈRE occurrence d'un groupe est l'original (`None`); les suivantes
    pointent vers elle. L'ordre d'entrée fait donc foi, ce qui rend le
    résultat déterministe pour une liste de fichiers triée.

    Returns:
        Liste de la même longueur que `hashes`.
    """
    originals: list[int] = []
    duplicate_of: list[int | None] = []

    for i, h in enumerate(hashes):
        match = next((j for j in originals if hamming(hashes[j], h) <= threshold), None)
        duplicate_of.append(match)
        if match is None:
            originals.append(i)

    return duplicate_of


def hash_images(images: Iterable) -> list[int]:
    return [dhash(img) for img in images]


def partition(
    items: Sequence, duplicate_of: Sequence[int | None]
) -> tuple[list, list]:
    """Sépare (originaux, doublons) selon le résultat de `find_duplicates`."""
    uniques = [it for it, dup in zip(items, duplicate_of, strict=True) if dup is None]
    dupes = [it for it, dup in zip(items, duplicate_of, strict=True) if dup is not None]
    return uniques, dupes
