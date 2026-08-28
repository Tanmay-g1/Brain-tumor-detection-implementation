"""
Step 13: Multi-Class Integrated Gradients (SHAP alternative)
Computes attribution maps for each class
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from scipy.ndimage import gaussian_filter

os.makedirs("outputs/multiclass/shap", exist_ok=True)

MODEL_PATH   = "models/brain_tumor_multiclass.h5"
IMG_SIZE     = (224, 224)
CLASS_NAMES  = ['glioma', 'meningioma', 'pituitary', 'notumor']
NUM_CLASSES  = 4

model = load_model(MODEL_PATH)
print("✓ Model loaded: brain_tumor_multiclass.h5")

# ── Integrated Gradients ──────────────────────────────────────────────────────
def compute_integrated_gradients(model, img_batch, class_idx, num_steps=50):
    """Compute Integrated Gradients for a specific class."""
    baseline = tf.zeros_like(img_batch)
    integrated_grads = None
    
    for step in range(num_steps):
        alpha = tf.constant(step / num_steps, dtype=tf.float32)
        interpolated_img = baseline + alpha * (img_batch - baseline)
        interpolated_img = tf.Variable(interpolated_img)
        
        with tf.GradientTape() as tape:
            predictions = model(interpolated_img, training=False)
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, interpolated_img)
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
            img_batch_tf = tf.convert_to_tensor(np.expand_dims(img_01, 0), dtype=tf.float32)
            
            test_images[class_name] = {
                'img_01': img_01,
                'img_batch_tf': img_batch_tf,
                'path': img_path
            }
            print(f"  ✓ {class_name}: {fname}")

# ── Compute Integrated Gradients ──────────────────────────────────────────────
print(f"\nComputing Integrated Gradients (50 steps per image)...")

fig, axes = plt.subplots(NUM_CLASSES, 2, figsize=(10, 4*NUM_CLASSES))

for row, class_name in enumerate(CLASS_NAMES):
    if class_name not in test_images:
        print(f"  ⚠ Skipped {class_name}")
        continue
    
    print(f"  Computing {class_name}...")
    
    img_data = test_images[class_name]
    img_01 = img_data['img_01']
    img_batch_tf = img_data['img_batch_tf']
    
    # Get prediction
    pred = model.predict(img_batch_tf.numpy(), verbose=0)[0]
    pred_class_idx = np.argmax(pred)
    confidence = pred[pred_class_idx] * 100
    
    # Integrated Gradients for predicted class
    saliency = compute_integrated_gradients(model, img_batch_tf, pred_class_idx)
    
    # Plot
    # Column 1: Original
    axes[row, 0].imshow(img_01)
    axes[row, 0].set_title(f"{class_name.upper()}", fontsize=11)
    axes[row, 0].axis("off")
    
    # Column 2: Saliency map
    vmax = saliency.max()
    axes[row, 1].imshow(img_01, alpha=0.3)
    im = axes[row, 1].imshow(saliency, cmap="jet", vmin=0, vmax=vmax, alpha=0.8)
    pred_name = CLASS_NAMES[pred_class_idx]
    axes[row, 1].set_title(f"Integrated Gradients: {pred_name}\n({confidence:.1f}%)", fontsize=11)
    axes[row, 1].axis("off")
    plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)
    
    # Save individual
    fig_ind, ax_ind = plt.subplots(1, 2, figsize=(10, 4))
    ax_ind[0].imshow(img_01)
    ax_ind[0].set_title(f"Original: {class_name}", fontsize=11)
    ax_ind[0].axis("off")
    
    ax_ind[1].imshow(img_01, alpha=0.3)
    im_ind = ax_ind[1].imshow(saliency, cmap="jet", vmin=0, vmax=vmax, alpha=0.8)
    ax_ind[1].set_title(f"Integrated Gradients", fontsize=11)
    ax_ind[1].axis("off")
    plt.colorbar(im_ind, ax=ax_ind[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(f"outputs/multiclass/shap/ig_{class_name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig_ind)

plt.suptitle("Multi-Class Integrated Gradients: Attribution Maps", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/multiclass/ig_all.png", dpi=100, bbox_inches="tight")
plt.close()

print("✓ Saved: outputs/multiclass/ig_all.png")
print("✓ Next: python 14_mc_xai_combined.py\n")
