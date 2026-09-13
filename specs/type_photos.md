# Supported Photo Types (Views)

Each image must be labeled with **exactly one** view category.

## 📷 Allowed View Categories

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

## Source of truth
These ten categories are defined in code in
`claimsight/domain/taxonomy.py::VIEW_CLASSES`. That list governs; this document
describes it. Horizontal-flip behaviour (left ↔ right) lives alongside it in
`VIEW_MIRROR_MAP`.
