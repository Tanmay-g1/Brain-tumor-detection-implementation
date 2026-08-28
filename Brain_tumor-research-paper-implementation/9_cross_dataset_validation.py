"""
Step 9: Cross-Dataset Validation (Section 4, page 8 of paper)
Tests generalizability on a SECOND dataset:
  "Brain MRI Images for Brain Tumor Detection" by Navoneel Chakrabarty
  Paper achieved: 92% accuracy, 94% precision, 93% recall, 93% F1

SETUP: Download from Kaggle and place as:
  data/cross_val/
    yes/   (253 tumor images)
    no/    (98 non-tumor images)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

CROSS_DIR  = "data/cross_val"
MODEL_PATH = "models/brain_tumor_cnn.h5"
IMG_SIZE   = (150, 150)

# ── Check dataset ─────────────────────────────────────────────────────────────
if not os.path.exists(CROSS_DIR):
    print("=" * 55)
    print("  CROSS-VALIDATION DATASET NOT FOUND")
    print("=" * 55)
    print("  Download: 'Brain MRI Images for Brain Tumor Detection'")
    print("  Author  : Navoneel Chakrabarty (Kaggle)")
    print("  Place as:")
    print("    data/cross_val/yes/   (tumor images)")
    print("    data/cross_val/no/    (non-tumor images)")
    print("\n  Paper result: Acc=92%, Prec=94%, Rec=93%, F1=93%")
    exit(0)

model = load_model(MODEL_PATH)

# ── Load images ───────────────────────────────────────────────────────────────
def load_folder(folder):
    images = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                img = keras_image.load_img(
                    os.path.join(folder, fname), target_size=IMG_SIZE)
                images.append(keras_image.img_to_array(img) / 255.0)
            except Exception:
                pass
    return np.array(images)

print("Loading cross-validation images ...")
tumor_imgs   = load_folder(os.path.join(CROSS_DIR, "yes"))
notumor_imgs = load_folder(os.path.join(CROSS_DIR, "no"))
X = np.concatenate([tumor_imgs, notumor_imgs], axis=0)
y_true = np.array([1] * len(tumor_imgs) + [0] * len(notumor_imgs))
print(f"  Tumor: {len(tumor_imgs)}  |  No Tumor: {len(notumor_imgs)}  |  Total: {len(X)}")

# ── Predict ───────────────────────────────────────────────────────────────────
y_prob  = model.predict(X, batch_size=8, verbose=1).flatten()
y_pred  = (y_prob >= 0.5).astype(int)

# ── Metrics ───────────────────────────────────────────────────────────────────
acc  = accuracy_score(y_true,  y_pred)
prec = precision_score(y_true, y_pred)
rec  = recall_score(y_true,    y_pred)
f1   = f1_score(y_true,        y_pred)

print("\n" + "=" * 50)
print("   CROSS-DATASET VALIDATION RESULTS")
print("=" * 50)
print(f"  Accuracy  : {acc*100:.2f}%   (paper: 92%)")
print(f"  Precision : {prec*100:.2f}%  (paper: 94%)")
print(f"  Recall    : {rec*100:.2f}%   (paper: 93%)")
print(f"  F1 Score  : {f1*100:.2f}%    (paper: 93%)")
print("=" * 50)

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["No Tumor", "Tumor"],
            yticklabels=["No Tumor", "Tumor"])
plt.title("Confusion Matrix — Cross-Dataset Validation")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("outputs/cross_val_confusion_matrix.png", dpi=100)
plt.close()

# ── Bar chart: our model vs paper values ──────────────────────────────────────
metrics    = ["Accuracy", "Precision", "Recall", "F1-Score"]
paper_vals = [92, 94, 93, 93]
our_vals   = [round(acc*100,1), round(prec*100,1),
              round(rec*100,1),  round(f1*100,1)]

x, w = np.arange(4), 0.35
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, paper_vals, w, label="Paper result", color="#e74c3c")
b2 = ax.bar(x + w/2, our_vals,   w, label="Our model",    color="#2ecc71")
for bar, val in zip(list(b1)+list(b2), paper_vals+our_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{val}%", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim([80, 105]); ax.set_ylabel("Score (%)")
ax.set_title("Generalizability: Our Model vs Paper on Cross-Validation Dataset")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/cross_val_comparison.png", dpi=100)
plt.close()

print("Saved: outputs/cross_val_confusion_matrix.png")
print("Saved: outputs/cross_val_comparison.png")
