"""
Step 4: Grad-CAM — XAI Technique 1
Reproduces Fig. 12 from paper (page 13):
  "MRI brain scan (left) with corresponding Grad-CAM visualization (right)
   highlighting model focus areas for tumor detection"
Blue = lower tumor probability, Red = higher tumor probability
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/brain_tumor_cnn.h5"
IMG_SIZE   = (150, 150)
LAST_CONV  = "conv5_last"        # last conv layer name from 2_train_cnn.py
OUTPUT_DIR = "outputs/gradcam"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load_model(MODEL_PATH)
model(tf.zeros((1, 150, 150, 3)))  # build the model

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
"""def compute_gradcam(model, img_array):
    img_array: (1, 150, 150, 3) float32 in [0,1]
    Returns heatmap (150,150), tumor_probability
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(LAST_CONV).output, model.outputs]
    )
    with tf.GradientTape() as tape:
     inputs = tf.cast(img_array, tf.float32)
     tape.watch(inputs)

     conv_outputs, predictions = grad_model(inputs)
     predictions = predictions[0][:, 0]   # binary output neuron
     loss = predictions

    grads        = tape.gradient(loss, conv_outputs)          # (1, h, w, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))      # (C,)
    conv_outputs = conv_outputs[0]                            # (h, w, C)
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap).numpy()
    heatmap      = np.maximum(heatmap, 0)
    heatmap     /= (heatmap.max() + 1e-8)

    tumor_prob = float(predictions.numpy().flatten()[0])
    return heatmap, tumor_prob"""
def compute_gradcam(model, img_array):
    last_conv_layer = model.get_layer(LAST_CONV)

    # Split model into feature extractor + classifier head for stable gradients.
    feature_extractor = tf.keras.models.Model(
        inputs=model.inputs[0],
        outputs=last_conv_layer.output
    )

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    found_last_conv = False
    for layer in model.layers:
        if found_last_conv:
            x = layer(x)
        if layer.name == LAST_CONV:
            found_last_conv = True

    classifier_model = tf.keras.models.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(img_array, training=False)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs, training=False)
        loss = tf.reduce_mean(predictions[:, 0])   # scalar target for gradients

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError(
            f"Grad-CAM failed: gradients are None for layer '{LAST_CONV}'. "
            "Check that LAST_CONV points to a Conv2D layer connected to output."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)

    heatmap = heatmap.numpy()

    tumor_prob = float(predictions.numpy()[0][0])

    return heatmap, tumor_prob

def overlay_heatmap(original_01, heatmap):
    h, w = original_01.shape[:2]
    heatmap_r   = cv2.resize(heatmap, (w, h))
    heatmap_u8  = np.uint8(255 * heatmap_r)
    heatmap_col = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    original_u8 = np.uint8(255 * original_01)
    superimposed = cv2.addWeighted(original_u8, 0.55, heatmap_col, 0.45, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)


def load_image(img_path):
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img) / 255.0
    return arr, np.expand_dims(arr, 0).astype(np.float32)

# ── Run on tumor and non-tumor examples (matches Fig. 12) ─────────────────────
samples = {
    "Tumor":    os.path.join("data", "Testing", "yes"),
    "No Tumor": os.path.join("data", "Testing", "no")
}

fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))

for row, (label, folder) in enumerate(samples.items()):
    img_file = sorted(f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))[0]
    img_path = os.path.join(folder, img_file)

    original, img_batch = load_image(img_path)
    heatmap, tumor_prob = compute_gradcam(model, img_batch)
    overlay             = overlay_heatmap(original, heatmap)

    pred_label = "Tumor" if tumor_prob >= 0.5 else "No Tumor"

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"gradcam_{label.replace(' ', '_')}.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    axes[row, 0].imshow(original)
    axes[row, 0].set_title(f"Input MRI\n(True: {label})", fontsize=10)
    axes[row, 0].axis("off")

    axes[row, 1].imshow(heatmap, cmap="jet")
    axes[row, 1].set_title("Grad-CAM Heatmap", fontsize=10)
    axes[row, 1].axis("off")

    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title(
        f"Explanation\nPred: {pred_label} ({tumor_prob:.2%})", fontsize=10)
    axes[row, 2].axis("off")

plt.suptitle(
    "Grad-CAM: Red = higher tumor probability, Blue = lower\n"
    "(Reproduces Fig. 12 from paper)",
    fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig("outputs/gradcam_all.png", dpi=100, bbox_inches="tight")
plt.show()
plt.close()
print("Saved: outputs/gradcam_all.png")
print(f"Individual images saved in: {OUTPUT_DIR}/")
