"""
Step 5: LIME — XAI Technique 2
Reproduces Fig. 10 & 11 from paper (pages 12-13):
  Fig. 10: "Top row = tumor images, bottom row = no-tumor. Yellow outlines
            highlight the areas that most influenced the model's decision."
  Fig. 11: "Original MRI (left) and highlighting the presence of tumor (right)"
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "models/brain_tumor_cnn.h5"
IMG_SIZE     = (150, 150)
NUM_SAMPLES  = 1000     # LIME perturbation samples
NUM_FEATURES = 10       # superpixels to highlight
OUTPUT_DIR   = "outputs/lime"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load_model(MODEL_PATH)

# ── Prediction wrapper ────────────────────────────────────────────────────────
# LIME expects predict_fn to return shape (N, num_classes)
# Binary sigmoid: convert to [P(no tumor), P(tumor)]
def predict_fn(images):
    images_scaled = images.astype(np.float32) / 255.0
    p_tumor = model.predict(images_scaled, verbose=0).flatten()
    return np.column_stack([1 - p_tumor, p_tumor])   # shape (N, 2)

explainer = lime_image.LimeImageExplainer(random_state=42)

def load_image_uint8(img_path):
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    return keras_image.img_to_array(img).astype(np.uint8)

# ── Samples: multiple tumor + no-tumor images (matches Fig. 10) ───────────────
SAMPLES_PER_CLASS = 5

tumor_dir    = os.path.join("data", "Testing", "yes")
notumor_dir  = os.path.join("data", "Testing", "no")
tumor_files  = sorted(os.listdir(tumor_dir))[:SAMPLES_PER_CLASS]
notumor_files= sorted(os.listdir(notumor_dir))[:SAMPLES_PER_CLASS]

fig, axes = plt.subplots(2, SAMPLES_PER_CLASS, figsize=(20, 8))

for col, fname in enumerate(tumor_files):
    img_path = os.path.join(tumor_dir, fname)
    img_u8   = load_image_uint8(img_path)

    exp = explainer.explain_instance(
        img_u8, predict_fn, top_labels=2,
        hide_color=0, num_samples=NUM_SAMPLES, random_seed=42)

    tumor_label = 1   # index 1 = tumor class
    temp, mask = exp.get_image_and_mask(
        tumor_label, positive_only=True,
        num_features=NUM_FEATURES, hide_rest=False)
    axes[0, col].imshow(mark_boundaries(temp / 255.0, mask, color=(1, 1, 0)))
    axes[0, col].set_title(f"Tumor", fontsize=9)
    axes[0, col].axis("off")

for col, fname in enumerate(notumor_files):
    img_path = os.path.join(notumor_dir, fname)
    img_u8   = load_image_uint8(img_path)

    exp = explainer.explain_instance(
        img_u8, predict_fn, top_labels=2,
        hide_color=0, num_samples=NUM_SAMPLES, random_seed=42)

    notumor_label = 0
    temp, mask = exp.get_image_and_mask(
        notumor_label, positive_only=True,
        num_features=NUM_FEATURES, hide_rest=False)
    axes[1, col].imshow(mark_boundaries(temp / 255.0, mask, color=(1, 1, 0)))
    axes[1, col].set_title(f"No Tumor", fontsize=9)
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Tumor", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("No Tumor", fontsize=12, fontweight="bold")

plt.suptitle(
    "LIME Explanations — Yellow outlines show regions that most influenced prediction\n"
    "(Reproduces Fig. 10 from paper)",
    fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig("outputs/lime_grid.png", dpi=100, bbox_inches="tight")
plt.show()
plt.close()
print("Saved: outputs/lime_grid.png")

# ── Single image detail view (matches Fig. 11) ────────────────────────────────
img_path = os.path.join(tumor_dir, tumor_files[0])
img_u8   = load_image_uint8(img_path)

exp = explainer.explain_instance(
    img_u8, predict_fn, top_labels=2,
    hide_color=0, num_samples=NUM_SAMPLES, random_seed=42)

temp_pos,  mask_pos  = exp.get_image_and_mask(1, positive_only=True,
                                               num_features=NUM_FEATURES, hide_rest=False)
temp_neg,  mask_neg  = exp.get_image_and_mask(1, positive_only=False,
                                               negative_only=True,
                                               num_features=NUM_FEATURES, hide_rest=False)
temp_both, mask_both = exp.get_image_and_mask(1, positive_only=False,
                                               num_features=NUM_FEATURES, hide_rest=False)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
axes[0].imshow(img_u8);                                                         axes[0].set_title("Input MRI")
axes[1].imshow(mark_boundaries(temp_pos  / 255.0, mask_pos,  color=(1,1,0))); axes[1].set_title("Explanation\n(Positive regions)")
axes[2].imshow(mark_boundaries(temp_neg  / 255.0, mask_neg,  color=(1,0,0))); axes[2].set_title("Negative regions")
axes[3].imshow(mark_boundaries(temp_both / 255.0, mask_both));                 axes[3].set_title("Both (+/-)")
for ax in axes: ax.axis("off")
plt.suptitle("LIME Detail — Reproduces Fig. 11 from paper", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/lime_detail.png", dpi=100, bbox_inches="tight")
plt.show()
plt.close()
print("Saved: outputs/lime_detail.png")
print(f"Individual LIME images saved in: {OUTPUT_DIR}/")
