import os

# mapping ancien_id -> nouveau_id
PART_CLASSES = {
    5: "front bumper",
    6: "front left door",
    7: "front right door",
    10: "headlights",
    11: "hood",
    14: "left fender",
    20: "rear bumper",
    21: "rear left door",
    22: "rear left fender",
    23: "rear right door",
    24: "rear right fender",
    25: "rear windshield",
    29: "right fender",
    33: "taillights",
    34: "trunk",
    36: "wheel/rim",
    37: "windshield",
}

NEW_ID = {old: i for i, old in enumerate(PART_CLASSES.keys())}

LABELS_DIR = "dataset/parts_yolo/labels"

for fname in os.listdir(LABELS_DIR):
    path = os.path.join(LABELS_DIR, fname)
    new_lines = []

    with open(path) as f:
        for line in f:
            cls = int(line.split()[0])
            if cls in PART_CLASSES:
                parts = line.split()
                parts[0] = str(NEW_ID[cls])
                new_lines.append(" ".join(parts))

    if new_lines:
        with open(path, "w") as f:
            f.write("\n".join(new_lines))
    else:
        os.remove(path)  # option simple V1
