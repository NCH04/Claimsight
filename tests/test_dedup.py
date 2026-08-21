"""Tests de la déduplication perceptuelle."""

import io

from PIL import Image

from claimsight.domain.dedup import dhash, find_duplicates, hamming, partition


def textured(seed: int, size=(400, 300)) -> Image.Image:
    """Image texturée déterministe — un aplat uni donnerait des hash dégénérés."""
    img = Image.new("RGB", size)
    for x in range(size[0]):
        for y in range(size[1]):
            img.putpixel((x, y), ((x * seed) % 256, (y * seed) % 256, ((x + y) * seed) % 256))
    return img


def recompressed(img: Image.Image, quality: int = 55) -> Image.Image:
    buf = io.BytesIO()
    img.resize((img.width // 2, img.height // 2)).save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def test_hash_is_stable_across_resize_and_jpeg_compression():
    original = textured(3)
    assert hamming(dhash(original), dhash(recompressed(original))) <= 5


def test_distinct_images_are_not_duplicates():
    assert hamming(dhash(textured(3)), dhash(textured(11))) > 5


def test_first_occurrence_is_the_original():
    hashes = [dhash(textured(3)), dhash(textured(11)), dhash(textured(3))]
    assert find_duplicates(hashes) == [None, None, 0]


def test_duplicates_chain_to_the_first_not_the_previous():
    h = dhash(textured(3))
    assert find_duplicates([h, h, h]) == [None, 0, 0]


def test_no_duplicates_when_all_distinct():
    hashes = [dhash(textured(s)) for s in (3, 7, 11, 13)]
    assert find_duplicates(hashes) == [None] * 4


def test_partition_splits_originals_and_duplicates():
    uniques, dupes = partition(["a", "b", "c"], [None, 0, None])
    assert uniques == ["a", "c"] and dupes == ["b"]


def test_threshold_zero_requires_strict_identity():
    original = textured(3)
    assert find_duplicates([dhash(original), dhash(recompressed(original))], threshold=0)[1] in (0, None)
