# Annotation Guidelines – Vehicle Damage Pipeline (v1)

## 1. Purpose
This document defines how images of damaged vehicles must be annotated so they can be used to train the models of the pipeline (view classification, part detection, damage detection, severity estimation).  
We are using **public datasets** as a starting point, so some labels will have to be **mapped** to our internal taxonomy.

---

## 2. What to annotate

We annotate up to **four levels**. If the information is not visible, we skip the level.

1. **View (required if inferable)**  
   One of:
   - `front`
   - `rear`
   - `left`
   - `right`
   - `front-left`
   - `front-right`
   - `rear-left`
   - `rear-right`
   - `closeup` *(use when the image is a zoom on a damaged area and the global orientation cannot be determined)*
   - `out_of_scope` *(use when the view cannot be determined)*

2. **Vehicle parts (if visible)**  
   Use the exact names from `supported_parts_v1.md`:
   - `front bumper`
   - `rear bumper`
   - `hood`
   - `trunk`
   - `windshield`
   - `front left door`
   - `front right door`
   - `rear left door`
   - `rear right door`
   - `left fender`
   - `right fender`
   - `headlights`
   - `taillights`

   Annotate the part **only if it is clearly present and identifiable** in the image.

3. **Damage type (if a part is damaged)**  
   Allowed values (keep it small for v1):
   - `scratch`
   - `dent`
   - `crack`
   - `broken_glass`
   - `deformation_impact`
   - `missing_part`

   If the public dataset has more detailed names, we map them (see §6).

4. **Severity (optional / best effort)**  
   - `minor` → superficial, small area, paint-level
   - `moderate` → visible deformation but still functional
   - `severe` → heavy deformation, broken glass, panel unusable

If the image does **not** allow a confident severity, leave it **unlabeled**. Do **not** invent.

---

## 3. General rules

1. **One image = one view**  
   Even if multiple parts are visible, the view describes the camera position, not the parts.

2. **Annotate only what is visible**  
   If the rear bumper is not in frame, don’t add it “because it should be there”.

3. **Closeup rule**  
   If the image is a zoom on a damaged area and you cannot tell if it’s the front or rear → set view to `closeup` and annotate only the part/damage.

4. **One part, multiple damages**  
   If a part has several damages, pick the **most severe** one for v1. We keep it simple.

5. **Naming must match JSON**  
   The value you use here must be the same that will appear later in `damaged_parts[].name`.

---

## 4. Annotation format

We aim to export to **COCO** or **YOLO** depending on the tool, but the semantics stay the same:

- **View** can be stored either:
  - as an image-level attribute (preferred), or
  - as a special label on a dummy box (fallback if the tool doesn’t support image attributes).

- **Parts and damages** are stored as bounding boxes (or masks if the dataset already has segmentation).

Example (conceptual):

```json
{
  "image": "VEH_0001_01.jpg",
  "view": "front-left",
  "objects": [
    {
      "category": "front bumper",
      "bbox": [120, 240, 420, 520],
      "damage": "deformation/impact",
      "severity": "moderate"
    },
    {
      "category": "headlights",
      "bbox": [180, 200, 240, 260],
      "damage": "crack/broken_glass",
      "severity": "severe"
    }
  ]
}
```

---

## 5. Source of truth

This document is prose; the code governs. Every list above is defined in
`claimsight/domain/taxonomy.py`:

| Concept  | Constant          | Count |
|----------|-------------------|-------|
| Views    | `VIEW_CLASSES`    | 10    |
| Damages  | `DAMAGE_LABELS`   | 6     |
| Severity | `SEVERITY_CLASSES`| 4     |
| Parts    | `PART_CLASSES`    | 15    |

Never redefine a class list anywhere else. Doing so is what previously let this
document and the dataset scripts drift apart (15 parts here, 17 in the code).

`unknown` is not in any list: it is the abstention value returned when a model
is unavailable or its confidence falls under the configured threshold.

---

## 6. Mapping public datasets

Public datasets rarely use our vocabulary, so each source gets an explicit
mapping file under `configs/`, keyed by the source's own class ids:

- `configs/source_part_ids_carparts_seg.json` — Carparts Segmentation
  (Ultralytics, CC BY 4.0). Covers 11 of our 15 parts; the four `fender`
  classes have no equivalent and stay unlabelled until annotated in-house.

Rules for a mapping file:

1. Every target name must exist in `PART_CLASSES`, or in `EXCLUDED_PARTS_V1`
   to be dropped on purpose rather than silently.
2. Several source classes may collapse into one target (`front_left_light`,
   `front_right_light` and `front_light` all become `headlights`).
3. Apply it with `scripts/remap_yolo_parts.py`, which reads the source
   read-only, writes elsewhere, and reports every box it dropped and why.
   Run it with `--dry_run` first.
4. Record the source's licence in the file's `_source` block — it decides
   what may be redistributed. See `dataset/README.md`.

