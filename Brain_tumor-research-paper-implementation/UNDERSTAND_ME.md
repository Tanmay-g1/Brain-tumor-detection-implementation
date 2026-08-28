Project: Brain Tumor CNN — Technical Breakdown
=============================================

This file contains a complete, end-to-end technical breakdown of the project so you can study or copy-paste for technical interview prep. It follows the requested structure: Project Overview, Data Preprocessing, Model Architecture, Training Setup, Evaluation & Inference, and a Full Runnable Script.

1) Project Overview & Folder Structure
--------------------------------------

- Purpose: Binary and multi-class brain MRI classification using custom CNNs with explainability (Grad-CAM, LIME, SHAP).
- Typical pipeline: prepare data -> train CNN -> evaluate -> generate XAI visualizations.

- Top-level layout (key files):
  - `1_preprocessing.py` — create ImageDataGenerators, sample grid.
  - `2_train_cnn.py` — model definition (improved CNN), training loop, callbacks.
  - `3_evaluate.py` — test evaluation & ROC/Confusion Matrix for binary model.
  - `4_gradcam.py`, `11_mc_gradcam.py` — Grad-CAM explanations (binary & multiclass).
  - `5_lime_explain.py` — LIME explanations.
  - `quick_eval.py` — quick evaluation for multiclass model.
  - `setup_dataset.py` — helper to produce `data/multiclass/` from Kaggle archive.
  - `models/` — saved .h5 model files.
  - `data/` — `Training/` and `Testing/` folders for each class.
  - `outputs/` — saved plots and metrics.

2) Data Preprocessing & Pipeline
--------------------------------

- Data loader: Keras `ImageDataGenerator` + `flow_from_directory` for directory-structured images.
- Target image sizes used in the repository:
  - baseline: `(150, 150)` (paper baseline)
  - improved: `(224, 224)` (used in `1_preprocessing.py`, `2_train_cnn.py`, and multiclass scripts)

- Normalization: `rescale=1.0/255` applied on all ImageDataGenerators. LIME predict wrapper also divides by 255.

- Training augmentation (from `2_train_cnn.py`):
```python
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1
)
```

- Example generator creation:
```python
train_gen = train_datagen.flow_from_directory(
    "data/Training",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=42
)

val_gen = train_datagen.flow_from_directory(
    "data/Training",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=42
)
```

- OpenCV image operations (Grad-CAM & overlays):
  - `cv2.resize` to scale heatmap to image dimensions
  - `cv2.applyColorMap` to colorize heatmap
  - `cv2.addWeighted` to blend colormap with original image

Overlay example (from `4_gradcam.py`):
```python
heatmap_r   = cv2.resize(heatmap, (w, h))
heatmap_u8  = np.uint8(255 * heatmap_r)
heatmap_col = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
original_u8 = np.uint8(255 * original_01)
superimposed = cv2.addWeighted(original_u8, 0.55, heatmap_col, 0.45, 0)
overlay_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
```

3) Model Architecture
----------------------

Two versions are in the repo: the paper baseline and an improved architecture used in `2_train_cnn.py`.

- Paper baseline (described in file header):
  - Input: 150 × 150 × 3
  - conv1: 32 filters, 3×3, ReLU, padding='same' → MaxPool(2,2)
  - conv2: 64 filters, 3×3, ReLU, padding='same' → MaxPool(2,2)
  - conv3: 128 filters, 3×3, ReLU, padding='same' → MaxPool(2,2)
  - conv4: 256 filters, 3×3, ReLU, padding='same' → MaxPool(2,2)
  - conv5: 256 filters, 3×3, ReLU, padding='same' → MaxPool(2,2)
  - Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(1, Sigmoid)

- Improved model (exact Keras code in `2_train_cnn.py`, copied here):
```python
from tensorflow.keras import layers, models

def build_improved_cnn(input_shape=(224,224,3)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=input_shape, name="conv1"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation="relu", padding="same", name="conv2"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation="relu", padding="same", name="conv3"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(256, (3,3), activation="relu", padding="same", name="conv4"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(256, (3,3), activation="relu", padding="same", name="conv5_last"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ], name="BrainTumor_CNN_Improved")
    return model
```

- Notes:
  - Convolution blocks follow `Conv2D -> BatchNormalization -> Dropout(0.25) -> MaxPooling2D`.
  - Classifier head uses dense layers with BatchNorm and Dropout(0.5) to reduce overfitting.
  - Final activation is `sigmoid` producing a scalar probability for binary classification.

4) Training Setup
-------------------

- Key hyperparameters (from `2_train_cnn.py`):
  - Optimizer: `Adam(learning_rate=1e-4)`
  - Loss: `binary_crossentropy`
  - Metrics: `accuracy`
  - Epochs: `EPOCHS = 30`
  - Batch size: `BATCH_SIZE = 8`

- Callbacks used:
  - `ModelCheckpoint` → save best model by `val_accuracy`.
  - `EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)`.
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)`.
  - `CSVLogger` to `outputs/training_log.csv`.

- Class imbalance mitigation: compute class weights with sklearn `compute_class_weight('balanced', ...)` and pass `class_weight` to `model.fit`.

- Training call example:
```python
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=cb_list,
    class_weight=class_weight_dict
)
```

5) Evaluation & Inference
--------------------------

- Binary model evaluation (key steps from `3_evaluate.py`):
  - Load saved model with `load_model("models/brain_tumor_cnn.h5")`.
  - Create test generator: `ImageDataGenerator(rescale=1./255).flow_from_directory(...)` with same `target_size`.
  - Get predicted probabilities: `y_prob = model.predict(test_gen).flatten()`.
  - Convert to labels: `y_pred = (y_prob >= 0.5).astype(int)`.
  - Compute metrics: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`.
  - ROC-AUC: `fpr, tpr, _ = roc_curve(y_true, y_prob); roc_auc = auc(fpr, tpr)`.

- Multi-class evaluation (from `quick_eval.py`): predict `model.predict(test_gen)` → `argmax` → `classification_report(..., output_dict=True)` and save CSV.

- Single-image inference example (used by Grad-CAM / LIME):
```python
img_pil = keras_image.load_img(img_path, target_size=IMG_SIZE)
img_arr = keras_image.img_to_array(img_pil) / 255.0
img_batch = np.expand_dims(img_arr, 0).astype(np.float32)
pred = model.predict(img_batch)  # sigmoid for binary
probability = float(pred[0][0])
label = "Tumor" if probability >= 0.5 else "No Tumor"
```

- Grad-CAM computation (repo approach):
  - Partition the Keras model into a feature extractor (inputs → last_conv_layer.output) and classifier head (input shape matching conv output → full outputs).
  - Use `tf.GradientTape()` to compute gradients of the target class score w.r.t conv feature maps.
  - Compute channel-wise pooled gradients, weight conv feature maps, sum, relu-clamp, and normalize to form heatmap.

6) Full Runnable Script (combined)
----------------------------------

Below is a single-file version that reproduces the typical workflow (train improved model, save, evaluate). It is a direct combination of code from the repository and is ready to copy-paste as `train_and_eval_combined.py`.

```python
# train_and_eval_combined.py
import os, json, numpy as np
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report

DATA_DIR   = "data"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 8
EPOCHS     = 30
MODEL_PATH = "models/brain_tumor_cnn_improved.h5"
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    zoom_range=0.2, horizontal_flip=True, validation_split=0.1
)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "Training"), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", subset="training", shuffle=True, seed=42
)
val_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "Training"), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", subset="validation", shuffle=False, seed=42
)

def build_improved_cnn(input_shape=(224,224,3)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=input_shape),
        layers.BatchNormalization(), layers.Dropout(0.25), layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(), layers.Dropout(0.25), layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(), layers.Dropout(0.25), layers.MaxPooling2D((2,2)),
        layers.Conv2D(256, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(), layers.Dropout(0.25), layers.MaxPooling2D((2,2)),
        layers.Conv2D(256, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(), layers.Dropout(0.25), layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"), layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(64, activation="relu"), layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

model = build_improved_cnn()
model.compile(optimizer=optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

cb_list = [
    callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    callbacks.CSVLogger("outputs/training_log.csv")
]

class_weights = compute_class_weight('balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=cb_list, class_weight=class_weight_dict)

with open("outputs/history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

# Evaluate on test set
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_gen = test_datagen.flow_from_directory(os.path.join(DATA_DIR,"Testing"), target_size=IMG_SIZE, batch_size=8, class_mode="binary", shuffle=False)

y_prob = model.predict(test_gen, verbose=1).flatten()
y_pred = (y_prob >= 0.5).astype(int)
y_true = test_gen.classes

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec  = recall_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred)
cm   = confusion_matrix(y_true, y_pred)
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

print("Test accuracy:", acc)
print("Precision, Recall, F1:", prec, rec, f1)
print("Confusion matrix:\n", cm)
print("ROC-AUC:", roc_auc)

model.save("models/final_saved_model.h5")
print("Saved final model at models/final_saved_model.h5")
```

7) Notes & Next Steps
---------------------

- Ensure `data/Training` and `data/Testing` follow expected layout: `data/Training/yes|no` for binary or `data/multiclass/Training/<class>` for multi-class.
- Install dependencies listed in the repository `requirements.txt` prior to running.
- If you want, I can append Grad-CAM and LIME code into this combined script or create an explicit `requirements_minimal.txt` listing exact versions.

----

End of file.
