# Photo coverage and view categories

**V1 ships photo coverage, not view classification.** An image is labelled with
the **faces it documents** — front, rear, left, right — and a diagonal shot
documents two. That is the multi-label `coverage` model, and the section at the
bottom of this page describes it.

The ten exclusive view categories below are kept for **description only**: the
output schema still carries a `view_prediction` field, and `VIEW_TO_FACES`
converts a legacy view-labelled dataset into coverage labels. No view model is
trained in V1.

## 📷 View categories (descriptive)

- front  
- rear  
- left  
- right  
- front-left  
- front-right  
- rear-left  
- rear-right  
- closeup  
- out_of_scope  

## ✔ Definition of Each View

### front
Vehicle photographed mainly from the front.

### rear
Vehicle photographed mainly from the rear.

### left
Driver-side view of the car (side profile).

### right
Passenger-side view of the car.

### front-left
Angled view between “front” and “left”.

### front-right
Angled view between “front” and “right”.

### rear-left
Angled view between "rear" and "left".

### rear-right
Angled view between "rear" and "right".

### closeup
Zoomed view focusing on a single part or damage area.
Used when the framing is too tight to determine the vehicle's orientation.
Excluded from view-model training (see `claimsight/training/tasks.py`).

### out_of_scope
The image cannot be assigned a view: no vehicle, unusable framing, or an
unrelated subject. Label it `out_of_scope` in Label Studio rather than guessing.

---

## Photo coverage — the shipped feature

The product question is not *which of ten views is this?* but **which of the
four faces does this claim document?** The ten categories above collapse onto
four faces, a diagonal counting for two:

| Face | Documented by |
|---|---|
| front | `front`, `front-left`, `front-right` |
| rear | `rear`, `rear-left`, `rear-right` |
| left | `left`, `front-left`, `rear-left` |
| right | `right`, `front-right`, `rear-right` |

The `coverage` model therefore predicts the four faces **independently**
(multi-label, one sigmoid per face) rather than picking one of ten exclusive
classes. One photo can document two faces, which is what actually happens with
a diagonal shot and what exclusive classes cannot express.

Annotating coverage means ticking one or two boxes — `scripts/annotate_coverage.py`
does it from the keyboard. The ten-view taxonomy remains supported for
descriptive purposes, and `VIEW_TO_FACES` converts an existing view-labelled
dataset into coverage labels without re-annotating.

## Source of truth
These ten categories are defined in code in
`claimsight/domain/taxonomy.py::VIEW_CLASSES`, the four faces in
`COVERAGE_FACES`, and the mapping between them in `VIEW_TO_FACES`. That list governs; this document
describes it. Horizontal-flip behaviour (left ↔ right) lives alongside it in
`VIEW_MIRROR_MAP`.
