"""
Step 3: Model Evaluation
Reproduces paper results (Section 4, page 8):
  Accuracy : 100% (train) / 98.67% (val)
  Precision: 98.50%
  Recall   : 98.50%
  F1 Score : 98.50%
  Confusion matrix: 165 TN, 131 TP, 2 FP, 2 FN
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, accuracy_score,
                             precision_score, recall_score, f1_score)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Load model ────────────────────────────────────────────────────────────────
model = load_model("models/brain_tumor_cnn.h5")

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = test_datagen.flow_from_directory(
    "data/Testing",
    target_size=(150, 150),
    batch_size=8,
    class_mode="binary",
    shuffle=False
)

CLASS_NAMES = list(test_gen.class_indices.keys())   # ['no', 'yes']
print(f"Classes: {CLASS_NAMES}  (0=no tumor, 1=tumor)")

# ── Predictions ───────────────────────────────────────────────────────────────
y_prob = model.predict(test_gen, verbose=1).flatten()   # sigmoid output, shape (N,)
y_pred = (y_prob >= 0.5).astype(int)                    # threshold at 0.5
y_true = test_gen.classes

# ── Metrics (paper reports these) ────────────────────────────────────────────
acc  = accuracy_score(y_true,  y_pred)
prec = precision_score(y_true, y_pred)
rec  = recall_score(y_true,    y_pred)
f1   = f1_score(y_true,        y_pred)

print("\n" + "=" * 50)
print("       PERFORMANCE METRICS")
print("=" * 50)
print(f"  Accuracy  : {acc*100:.2f}%   (paper: 98.67% val / 100% train)")
print(f"  Precision : {prec*100:.2f}%  (paper: 98.50%)")
print(f"  Recall    : {rec*100:.2f}%   (paper: 98.50%)")
print(f"  F1 Score  : {f1*100:.2f}%    (paper: 98.50%)")
print("=" * 50)

print("\n── Full Classification Report ──")
print(classification_report(y_true, y_pred,
                             target_names=["No Tumor", "Tumor"]))

# ── Performance bar chart (matches Fig. 4 in paper) ──────────────────────────
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values  = [acc * 100, prec * 100, rec * 100, f1 * 100]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(metrics, values, color="#2980b9", edgecolor="white")
for bar, val in zip(bars, values):
    ax.text(bar.get_width() - 1.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%", va="center", ha="right",
            color="white", fontweight="bold", fontsize=11)
ax.set_xlim([0, 105])
ax.set_xlabel("Performance Value (%)")
ax.set_title("Performance Metrics", fontsize=13)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/performance_metrics.png", dpi=100)
plt.close()
print("Saved: outputs/performance_metrics.png")

# ── Confusion matrix (matches Fig. 5 in paper) ───────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Tumor (0)", "Tumor (1)"],
            yticklabels=["No Tumor (0)", "Tumor (1)"],
            linewidths=1, annot_kws={"size": 14})
plt.title("Confusion Matrix\n(Paper: TN=165, TP=131, FP=2, FN=2)", fontsize=12)
plt.ylabel("True Label"); plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=100)
plt.close()
print("Saved: outputs/confusion_matrix.png")
print(f"\nConfusion Matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")
print("Paper values: TN=165, FP=2, FN=2, TP=131")

# ── ROC curve ────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc     = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([0, 1]); plt.ylim([0, 1.02])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Brain Tumor Detection")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/roc_auc.png", dpi=100)
plt.close()
print(f"Saved: outputs/roc_auc.png  (AUC = {roc_auc:.4f})")
