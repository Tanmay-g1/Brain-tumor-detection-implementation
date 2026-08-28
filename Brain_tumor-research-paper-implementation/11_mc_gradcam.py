"""
Step 11: Multi-Class Grad-CAM Visualization
Shows which regions influence predictions for each class
Layout: 4 rows (one per class) × 3 columns (Original | Heatmap | Overlay)
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

os.makedirs("outputs/multiclass/gradcam", exist_ok=True)

MODEL_PATH   = "models/brain_tumor_multiclass.h5"
IMG_SIZE     = (224, 224)
LAST_CONV    = "conv5"
CLASS_NAMES  = ['glioma', 'meningioma', 'pituitary', 'notumor']
NUM_CLASSES  = 4

model = load_model(MODEL_PATH)
print("✓ Model loaded: brain_tumor_multiclass.h5")

# ── Grad-CAM function ─────────────────────────────────────────────────────────
def compute_gradcam(model, img_array, class_idx):
    """Compute Grad-CAM for a specific class."""
    last_conv_layer = model.get_layer(LAST_CONV)
    
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
        loss = predictions[:, class_idx]  # Target class
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    
    heatmap = heatmap.numpy()
    
    # Resize and overlay
    h, w = IMG_SIZE
    hm_r = cv2.resize(heatmap, (w, h))
    colored = cv2.applyColorMap(np.uint8(255 * hm_r), cv2.COLORMAP_JET)
    orig_u8 = np.uint8(255 * img_array[0])
    sup = cv2.addWeighted(orig_u8, 0.55, colored, 0.45, 0)
    
    return cv2.cvtColor(sup, cv2.COLOR_BGR2RGB), heatmap

# ── Load sample images from each class ─────────────────────────────────────────
print("\nLoading sample images from test set...")

test_images = {}
test_paths = {}

for class_idx, class_name in enumerate(CLASS_NAMES):
    class_dir = os.path.join("data/multiclass/Testing", class_name)
    
    if os.path.exists(class_dir):
        files = sorted(os.listdir(class_dir))[:1]  # 1 sample per class
        
        for fname in files:
            img_path = os.path.join(class_dir, fname)
            img_pil = keras_image.load_img(img_path, target_size=IMG_SIZE)
            img_01 = keras_image.img_to_array(img_pil) / 255.0
            img_batch = np.expand_dims(img_01, 0).astype(np.float32)
            
            test_images[class_name] = {
                'img_01': img_01,
                'img_batch': img_batch,
                'img_pil': img_pil
            }
            test_paths[class_name] = img_path
            print(f"  ✓ {class_name}: {fname}")

# ── Compute Grad-CAM for each class ───────────────────────────────────────────
print("\nComputing Grad-CAM for each class...")

fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(12, 4*NUM_CLASSES))

for row, class_name in enumerate(CLASS_NAMES):
    if class_name not in test_images:
        print(f"  ⚠ Skipped {class_name} (no images)")
        continue
    
    print(f"  Computing {class_name}...")
    
    img_data = test_images[class_name]
    img_01 = img_data['img_01']
    img_batch = img_data['img_batch']
    
    # Get predictions
    pred = model.predict(img_batch, verbose=0)
    pred_class_idx = np.argmax(pred[0])
    confidence = pred[0][pred_class_idx] * 100
    
    # Compute Grad-CAM for the predicted class
    gc_img, heatmap = compute_gradcam(model, img_batch, pred_class_idx)
    
    # Plot
    # Column 1: Original
    axes[row, 0].imshow(img_01)
    axes[row, 0].set_title(f"{class_name.upper()}\n(Ground Truth)", fontsize=11)
    axes[row, 0].axis("off")
    
    # Column 2: Heatmap
    axes[row, 1].imshow(heatmap, cmap="jet")
    axes[row, 1].set_title(f"Grad-CAM Heatmap", fontsize=11)
    axes[row, 1].axis("off")
    
    # Column 3: Overlay
    axes[row, 2].imshow(gc_img)
    pred_name = CLASS_NAMES[pred_class_idx]
    axes[row, 2].set_title(f"Predicted: {pred_name}\n({confidence:.1f}%)", fontsize=11)
    axes[row, 2].axis("off")

plt.suptitle("Multi-Class Grad-CAM: Influential Regions per Class", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/multiclass/gradcam_all.png", dpi=100, bbox_inches="tight")
plt.close()

print("✓ Saved: outputs/multiclass/gradcam_all.png")
print("✓ Next: python 12_mc_lime.py\n")
