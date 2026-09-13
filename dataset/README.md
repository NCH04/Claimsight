# Datasets — provenance and licences

Images are **not** committed to this repository (`.gitignore` excludes them).
This file records where the training data comes from and what each source
permits, because those terms — not this repository's MIT licence — govern the
data and any model weights derived from it.

> **Read this before publishing weights.** A licence that permits training does
> not automatically permit redistributing the resulting weights. When a source
> is non-commercial, treat the weights trained on it as non-commercial too
> unless its authors say otherwise.

---

## Parts detection — in use

**Carparts Segmentation** · **CC BY 4.0** · 3,833 images
<https://docs.ultralytics.com/datasets/segment/carparts-seg/>

Commercial use permitted **with attribution**. Downloaded automatically by
`ultralytics`; remapped to our taxonomy by
`scripts/remap_yolo_parts.py --mapping configs/source_part_ids_carparts_seg.json`.

Covers 11 of the 15 parts in `PART_CLASSES`. The four `fender` classes have no
equivalent in the source and carry no training data yet — their class ids are
reserved so the id contract stays stable when they are annotated in-house.
`rear windshield` and `wheel/rim` exist in the source but are dropped on
purpose (`EXCLUDED_PARTS_V1`).

*Attribution to reproduce wherever the model is distributed:*
> Carparts Segmentation dataset, Ultralytics, licensed under CC BY 4.0.

---

## Damage classification — candidate, not yet used

**CarDD** · **non-commercial research and education only** · 4,000 images
<https://cardd-ustc.github.io/>

The reference dataset for car damage detection: six categories (dent, scratch,
crack, glass shatter, tire flat, lamp broken), high resolution, COCO-format
masks. Faces and licence plates are already mosaicked by its authors.

Two consequences before adopting it:

1. **Access is gated.** A licensing form must be signed and emailed; there is
   no anonymous download.
2. **Non-commercial only**, and the underlying images come from Flickr and
   Shutterstock. Weights trained on it must not ship under a permissive
   licence, and the repository must not redistribute the images.

Its six categories map onto `DAMAGE_LABELS` only partially — `tire flat` and
`lamp broken` have no equivalent, and `deformation_impact` / `missing_part`
have no source. Decide that mapping before training.

---

## View classification — no public source

No public dataset labels vehicle photos with the eight orientations in
`VIEW_CLASSES`. This model is trained solely on in-house Label Studio
annotations, which is why `dataset/annotations/` is the one data directory
kept under version control.

---

## Personal data

Claim photos routinely contain licence plates, faces and VINs — personal data
under the GDPR. Blur them before any image leaves the processing pipeline, and
never commit an unredacted photo, sample or screenshot to this repository.
