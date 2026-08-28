"""
Step 1: Data Preprocessing
Dataset : BR35H (binary — tumor / no-tumor)
Paper   : "Utilizing Customized CNN for Brain Tumor Prediction with XAI", Heliyon 2024

Expected folder structure:
    data/
      Training/
        yes/   (1500 tumor MRI images)
        no/    (1500 non-tumor MRI images)
      Testing/
        yes/   (30 tumor images  — from BR35H pred/ folder)
        no/    (30 non-tumor images — from BR35H pred/ folder)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config (IMPROVED: 224×224 for better feature extraction) ──────────────────────────
DATA_DIR   = "data"
IMG_SIZE   = (224, 224)   # UPGRADED from 150×150 for improved accuracy
BATCH_SIZE = 8            # paper Algorithm 1: batch_size = 8

os.makedirs("outputs", exist_ok=True)

# ── Generators ────────────────────────────────────────────────────────────────
# Paper: normalize pixels to [0, 1] by dividing by 255
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.1      # paper: 90% train, 10% val
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "Training"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",       # binary: 0 = no tumor, 1 = tumor
    subset="training",
    shuffle=True,
    seed=42
)

val_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "Training"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=42
)

test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "Testing"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

CLASS_NAMES = list(train_gen.class_indices.keys())   # ['no', 'yes'] alphabetically

# ── Sanity check ──────────────────────────────────────────────────────────────
print("=" * 50)
print(f"Classes      : {CLASS_NAMES}  (0={CLASS_NAMES[0]}, 1={CLASS_NAMES[1]})")
print(f"Train samples: {train_gen.samples}   (paper: 2700)")
print(f"Val   samples: {val_gen.samples}     (paper: 300)")
print(f"Test  samples: {test_gen.samples}    (paper: 60)")
print(f"Image size   : {IMG_SIZE}")
print(f"Batch size   : {BATCH_SIZE}")
print("=" * 50)

# ── Sample grid ───────────────────────────────────────────────────────────────
images, labels = next(train_gen)
n = min(16, len(images))
fig, axes = plt.subplots(2, 8, figsize=(20, 5))
for i, ax in enumerate(axes.flat):
    if i < n:
        ax.imshow(images[i], cmap="gray" if images[i].shape[-1] == 1 else None)
        ax.set_title("Tumor" if labels[i] == 1 else "No Tumor", fontsize=8)
    ax.axis("off")
plt.suptitle("Sample Training Images from BR35H Dataset", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/sample_grid.png", dpi=100)
plt.close()
print("Saved: outputs/sample_grid.png")
