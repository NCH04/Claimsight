import os

import pandas as pd

CSV_PATH = "dataset/labels.csv"
SRC_DIR  = "dataset/images"
DST_DIR  = "dataset/images_mapped"

os.makedirs(DST_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df[["image", "view"]].dropna()

# index des fichiers disponibles
files = os.listdir(SRC_DIR)

# index par "suffixe" (ce qui vient après le dernier '/')
# et aussi par "queue" après le dernier '-' (utile pour -0043.JPEG)
by_basename = {f: f for f in files}
by_tail = {}
for f in files:
    tail = f.split("-")[-1]  # ex: 0048.JPEG ou Car_damages_300.png
    by_tail.setdefault(tail, []).append(f)

missing = 0
ambiguous = 0

for p in df["image"].tolist():
    want = os.path.basename(p)  # ex: 1fb0e351-0043.JPEG
    want_tail = want.split("-")[-1]  # ex: 0043.JPEG

    src = None

    # 1) match exact
    if want in by_basename:
        src = want
    else:
        # 2) match par tail (0043.JPEG)
        cands = by_tail.get(want_tail, [])
        if len(cands) == 1:
            src = cands[0]
        elif len(cands) > 1:
            ambiguous += 1

    if src is None:
        missing += 1
        continue

    src_path = os.path.join(SRC_DIR, src)
    dst_path = os.path.join(DST_DIR, want)

    if not os.path.exists(dst_path):
        os.symlink(os.path.abspath(src_path), dst_path)

print(f"Done. Missing: {missing}, Ambiguous: {ambiguous}, Total rows: {len(df)}")
