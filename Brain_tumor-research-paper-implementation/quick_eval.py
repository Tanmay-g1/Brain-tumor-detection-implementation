import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

# ── Load model ─────────────────────────────────────────────────────────────
print("Loading trained model...")
model = keras.models.load_model("models/brain_tumor_multiclass.h5")

# ── Prepare test data ─────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

test_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
).flow_from_directory(
    "data/multiclass/Testing/",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    classes={'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
)

# ── Evaluate ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("📊 EVALUATING ON TEST SET")
print("="*70)

test_loss, test_acc = model.evaluate(test_gen, verbose=0)

print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"   Test Loss: {test_loss:.4f}")

# Get predictions for per-class metrics
print("\nGenerating predictions for per-class metrics...")
test_gen.reset()
predictions = model.predict(test_gen, verbose=0)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

# Classification report
print("\n" + "="*70)
print("📋 PER-CLASS METRICS")
print("="*70)
report = classification_report(
    y_true, y_pred,
    target_names=['glioma', 'meningioma', 'notumor', 'pituitary'],
    digits=4
)
print(report)

# Save metrics
import pandas as pd
report_dict = classification_report(
    y_true, y_pred,
    target_names=['glioma', 'meningioma', 'notumor', 'pituitary'],
    output_dict=True
)
df_metrics = pd.DataFrame(report_dict).transpose()
df_metrics.to_csv("outputs/multiclass/metrics_improved.csv")

print(f"\n✓ Metrics saved: outputs/multiclass/metrics_improved.csv")
print(f"{'='*70}\n")
