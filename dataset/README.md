# Datasets — provenance and licences

Images are **not** committed to this repository (`.gitignore` excludes them).
This file records where the training data comes from and what each source
permits, because those terms — not this repository's Apache 2.0 licence — govern the
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

Covers all 12 parts in `PART_CLASSES`. The four `fender` classes were dropped
from V1 (`EXCLUDED_PARTS_V1`): the source has no equivalent, so they would have
shipped as permanently empty classes. `wheel/rim` went the other way and was
reinstated — a buckled rim is evidence of impact in its own right — but the
source annotates wheels inconsistently, often only the front one, which caps it
at 0.400 mAP50 and makes it the detector's weakest class.

`side door` and `rear windshield` stay excluded.

*Attribution to reproduce wherever the model is distributed:*
> Carparts Segmentation dataset, Ultralytics, licensed under CC BY 4.0.

---

## Damage and severity classification — in use

**Car Damaged Severity Detection** · **CC BY 4.0** · 3,016 images (v29)
<https://universe.roboflow.com/car-damaged-detection-e66m0/car-damaged-severity-detection>

```bash
export ROBOFLOW_API_KEY=...       # jamais en argument, jamais dans un fichier
make fetch-damage                 # refuse de télécharger hors licence autorisée
```

Commercial use permitted **with attribution**. Its class names encode damage
type *and* severity at once (`severe-deformation`), so one dataset feeds both
classifiers. Mapped by `configs/source_damage_ids.json`, then read twice, because the
classifier is trained on two domains and served on both:

- `scripts/derive_damage_labels.py` → one row per **whole photo**, keeping the
  worst finding by ordinal severity.
- `scripts/derive_damage_crops.py` → one row per **damage polygon**, cut to its
  bounding box with the same 8% margin `Detection.crop` uses at inference, plus
  the parts of intact vehicles as `none`.

`scripts/merge_label_csv.py` concatenates the two into `damage_mixed.csv`
(10,583 rows over 3,284 photos). Training on whole photos alone produced a
classifier that answered `none` on 59 of 60 intact cars but on only 44 of the
277 part crops taken from those same cars.

Two gaps, both recorded in the mapping file:

- `missing_part` has no source class in v29 (older versions had `detachment`).
- The dataset contains **only damaged vehicles**, so `none` examples are drawn
  from the parts dataset, whose vehicles are assumed intact. That assumption is
  weak supervision, not ground truth.

`flat-tire` is dropped on purpose: a flat tire is not body damage, and folding
it into `deformation_impact` would be wrong.

*Attribution to reproduce wherever the model is distributed:*
> Car Damaged Severity Detection, Roboflow Universe, licensed under CC BY 4.0.

---

## CarDD — evaluated, rejected

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
