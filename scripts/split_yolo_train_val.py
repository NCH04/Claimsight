import os
import random
import shutil

ROOT = "dataset/parts_yolo"
IMAGES_DIR = os.path.join(ROOT, "images")
LABELS_DIR = os.path.join(ROOT, "labels")

TRAIN_IMAGES = os.path.join(IMAGES_DIR, "train")
VAL_IMAGES   = os.path.join(IMAGES_DIR, "val")
TRAIN_LABELS = os.path.join(LABELS_DIR, "train")
VAL_LABELS   = os.path.join(LABELS_DIR, "val")

os.makedirs(TRAIN_IMAGES, exist_ok=True)
os.makedirs(VAL_IMAGES, exist_ok=True)
os.makedirs(TRAIN_LABELS, exist_ok=True)
os.makedirs(VAL_LABELS, exist_ok=True)

# Extensions possibles
exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# On ne garde que les images qui ont un label .txt correspondant
imgs = []
for f in os.listdir(IMAGES_DIR):
    if os.path.isdir(os.path.join(IMAGES_DIR, f)):
        continue
    _, ext = os.path.splitext(f)
    if ext not in exts:
        continue
    stem, _ = os.path.splitext(f)
    lab = os.path.join(LABELS_DIR, stem + ".txt")
    if os.path.exists(lab):
        imgs.append(f)

print(f"Images avec labels: {len(imgs)}")

random.seed(42)
random.shuffle(imgs)

val_ratio = 0.2
n_val = max(1, int(len(imgs) * val_ratio))
val_set = set(imgs[:n_val])
train_set = imgs[n_val:]

def move_pair(img_name, dst_img_dir, dst_lbl_dir):
    stem, _ = os.path.splitext(img_name)
    src_img = os.path.join(IMAGES_DIR, img_name)
    src_lbl = os.path.join(LABELS_DIR, stem + ".txt")
    shutil.move(src_img, os.path.join(dst_img_dir, img_name))
    shutil.move(src_lbl, os.path.join(dst_lbl_dir, stem + ".txt"))

for img in train_set:
    move_pair(img, TRAIN_IMAGES, TRAIN_LABELS)

for img in val_set:
    move_pair(img, VAL_IMAGES, VAL_LABELS)

print(f"Train: {len(train_set)}  Val: {len(val_set)}")
