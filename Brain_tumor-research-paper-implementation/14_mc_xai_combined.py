"""
Step 14: Multi-Class Combined XAI Visualization
Side-by-side: Original | Grad-CAM | LIME | Integrated Gradients
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from lime import lime_image
from skimage.segmentation import mark_boundaries

os.makedirs("outputs/multiclass/combined_xai", exist_ok=True)

MODEL_PATH   = "models/brain_tumor_multiclass.h5"
IMG_SIZE     = (224, 224)
CLASS_NAMES  = ['glioma', 'meningioma', 'pituitary', 'notumor']
NUM_CLASSES  = 4
LAST_CONV    = "conv5"

model = load_model(MODEL_PATH)
print("✓ Model loaded: brain_tumor_multiclass.h5")

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def compute_gradcam(model, img_batch, class_idx):
    """Compute Grad-CAM overlay."""
    last_conv_layer = model.get_layer(LAST_CONV)
    
    feature_extractor = tf.keras.models.Model(
        inputs=model.inputs[0], outputs=last_conv_layer.output
    )
    
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    found = False
    for layer in model.layers:
        if found:
            x = layer(x)
        if layer.name == LAST_CONV:
            found = True
    classifier_model = tf.keras.models.Model(classifier_input, x)
    
    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(img_batch, training=False)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs, training=False)
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    
    h, w = IMG_SIZE
    hm_r = cv2.resize(heatmap, (w, h))
    colored = cv2.applyColorMap(np.uint8(255 * hm_r), cv2.COLORMAP_JET)
    orig_u8 = np.uint8(255 * img_batch[0])
    sup = cv2.addWeighted(orig_u8, 0.55, colored, 0.45, 0)
    
    return cv2.cvtColor(sup, cv2.COLOR_BGR2RGB)

# ── LIME ───────────────────────────────────────────────────────────────────────
def predict_fn_lime(images):
    """Predict probabilities for LIME."""
    return model.predict(images.astype(np.float32) / 255.0, verbose=0)

def compute_lime(img_u8, class_idx, num_samples=500):
    """Compute LIME superpixel mask."""
    explainer = lime_image.LimeImageExplainer(random_state=42)
    exp = explainer.explain_instance(
        img_u8, predict_fn_lime, top_labels=NUM_CLASSES,
        hide_color=0, num_samples=num_samples, random_seed=42
    )
    temp, mask = exp.get_image_and_mask(
        class_idx, positive_only=True, num_features=10, hide_rest=False
    )
    return mark_boundaries(temp / 255.0, mask, color=(1, 1, 0))

# ── Integrated Gradients ──────────────────────────────────────────────────────
def compute_ig(model, img_batch, class_idx, num_steps=50):
    """Compute Integrated Gradients."""
    baseline = tf.zeros_like(img_batch)
    integrated_grads = None
    
    for step in range(num_steps):
        alpha = tf.constant(step / num_steps, dtype=tf.float32)
        interpolated = baseline + alpha * (img_batch - baseline)
        interpolated = tf.Variable(interpolated)
        
        with tf.GradientTape() as tape:
            predictions = model(interpolated, training=False)
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, interpolated)
        if grads is not None:
            if integrated_grads is None:
                integrated_grads = grads
            else:
                integrated_grads += grads
    
    if integrated_grads is None:
        integrated_grads = tf.zeros_like(img_batch)
    
    integrated_grads = (img_batch - baseline) * integrated_grads / num_steps
    saliency = tf.reduce_sum(tf.abs(integrated_grads[0]), axis=-1)
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-8)
    return saliency.numpy()

# ── Load sample images ────────────────────────────────────────────────────────
print("\nLoading sample images...")

test_images = {}

for class_idx, class_name in enumerate(CLASS_NAMES):
    class_dir = os.path.join("data/multiclass/Testing", class_name)
    
    if os.path.exists(class_dir):
        files = sorted(os.listdir(class_dir))[:1]
        
        for fname in files:
            img_path = os.path.join(class_dir, fname)
            img_pil = keras_image.load_img(img_path, target_size=IMG_SIZE)
            img_01 = keras_image.img_to_array(img_pil) / 255.0
            img_u8 = np.uint8(img_01 * 255)
            img_batch = np.expand_dims(img_01, 0).astype(np.float32)
            img_batch_tf = tf.convert_to_tensor(img_batch, dtype=tf.float32)
            
            test_images[class_name] = {
                'img_01': img_01,
                'img_u8': img_u8,
                'img_batch': img_batch,
                'img_batch_tf': img_batch_tf,
                'path': img_path
            }
            print(f"  ✓ {class_name}: {fname}")

# ── Compute Combined XAI ──────────────────────────────────────────────────────
print(f"\nComputing Combined XAI for each class...")

fig, axes = plt.subplots(NUM_CLASSES, 4, figsize=(16, 4*NUM_CLASSES))

for row, class_name in enumerate(CLASS_NAMES):
    if class_name not in test_images:
        print(f"  ⚠ Skipped {class_name}")
        continue
    
    print(f"  Processing {class_name}...")
    
    data = test_images[class_name]
    img_01 = data['img_01']
    img_u8 = data['img_u8']
    img_batch = data['img_batch']
    img_batch_tf = data['img_batch_tf']
    
    # Prediction
    pred = model.predict(img_batch, verbose=0)[0]
    pred_class_idx = np.argmax(pred)
    confidence = pred[pred_class_idx] * 100
    pred_name = CLASS_NAMES[pred_class_idx]
    
    # Column 0: Original
    axes[row, 0].imshow(img_01)
    axes[row, 0].set_title(f"{class_name.upper()}\n(Ground Truth)", fontsize=10)
    axes[row, 0].axis("off")
    
    # Column 1: Grad-CAM
    print(f"    - Grad-CAM...")
    gc = compute_gradcam(model, img_batch, pred_class_idx)
    axes[row, 1].imshow(gc)
    axes[row, 1].set_title(f"Grad-CAM", fontsize=10)
    axes[row, 1].axis("off")
    
    # Column 2: LIME
    print(f"    - LIME...")
    lime_img = compute_lime(img_u8, pred_class_idx)
    axes[row, 2].imshow(lime_img)
    axes[row, 2].set_title(f"LIME", fontsize=10)
    axes[row, 2].axis("off")
    
    # Column 3: Integrated Gradients
    print(f"    - Integrated Gradients...")
    ig = compute_ig(model, img_batch_tf, pred_class_idx)
    axes[row, 3].imshow(img_01, alpha=0.3)
    im = axes[row, 3].imshow(ig, cmap="RdBu_r", vmin=0, vmax=1, alpha=0.8)
    axes[row, 3].set_title(f"Integrated Gradients", fontsize=10)
    axes[row, 3].axis("off")
    plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)
    
    # Save individual class comparison
    fig_ind, ax_ind = plt.subplots(1, 4, figsize=(16, 4))
    
    ax_ind[0].imshow(img_01)
    ax_ind[0].set_title(f"Original: {class_name}", fontsize=11)
    ax_ind[0].axis("off")
    
    ax_ind[1].imshow(gc)
    ax_ind[1].set_title("Grad-CAM", fontsize=11)
    ax_ind[1].axis("off")
    
    ax_ind[2].imshow(lime_img)
    ax_ind[2].set_title("LIME", fontsize=11)
    ax_ind[2].axis("off")
    
    ax_ind[3].imshow(img_01, alpha=0.3)
    im_ind = ax_ind[3].imshow(ig, cmap="RdBu_r", vmin=0, vmax=1, alpha=0.8)
    ax_ind[3].set_title("Integrated Gradients", fontsize=11)
    ax_ind[3].axis("off")
    plt.colorbar(im_ind, ax=ax_ind[3], fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Prediction: {pred_name} ({confidence:.1f}%)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"outputs/multiclass/combined_xai/xai_{class_name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig_ind)

plt.suptitle("Multi-Class Combined XAI Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/multiclass/xai_combined_all.png", dpi=100, bbox_inches="tight")
plt.close()

print("✓ Saved: outputs/multiclass/xai_combined_all.png")
print("✓ Combined XAI analysis complete!")
print("✓ Next: python app.py (or re-run with different images)\n")
