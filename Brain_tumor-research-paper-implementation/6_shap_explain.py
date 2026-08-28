"""
Step 6: XAI Explanations — Gradient-based Saliency Maps
Reproduces Fig. 8 & 9 from paper (pages 11-12):
  Fig. 8: "Explanation visualizations for both tumor and non-tumor MRI images"
  Fig. 9: "MRI image (left) and its explanation (right)"
  Red areas = higher influence on tumor prediction
  Blue areas = lower influence on tumor prediction

NOTE: Uses gradient-based saliency maps (computationally efficient alternative)
      providing similar insights to SHAP with practical performance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH        = "models/brain_tumor_cnn.h5"
IMG_SIZE          = (150, 150)
OUTPUT_DIR        = "outputs/shap"

# ★★★ CONFIGURE NUMBER OF SAMPLES HERE ★★★
NUM_TUMOR_SAMPLES    = 3     # Change to 2, 3, 5, etc. for more tumor images
NUM_NOTUMOR_SAMPLES  = 3    # Change to 2, 3, 5, etc. for more non-tumor images
# ★★★ END CONFIG ★★★

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
from tensorflow.keras.models import load_model
model = load_model(MODEL_PATH)
print(f"Model loaded successfully")
print(f"Configuration: {NUM_TUMOR_SAMPLES} tumor sample(s) + {NUM_NOTUMOR_SAMPLES} non-tumor sample(s)")

# ── Load images with PIL ──────────────────────────────────────────────────────
def load_image_pil(img_path):
    """Load image using PIL and return normalized array."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    return np.array(img) / 255.0

# ── Load test images (multiple samples) ────────────────────────────────────────
test_tumor_dir   = os.path.join("data", "Testing", "yes")
test_notumor_dir = os.path.join("data", "Testing", "no")

# Load multiple tumor images
tumor_files = sorted(os.listdir(test_tumor_dir))[:NUM_TUMOR_SAMPLES]
tumor_imgs = []
tumor_labels = []
for i, fname in enumerate(tumor_files):
    try:
        img = load_image_pil(os.path.join(test_tumor_dir, fname))
        tumor_imgs.append(img)
        tumor_labels.append(f"Tumor {i+1}")
    except:
        print(f"  ⚠ Skipped: {fname}")

# Load multiple non-tumor images
notumor_files = sorted(os.listdir(test_notumor_dir))[:NUM_NOTUMOR_SAMPLES]
notumor_imgs = []
notumor_labels = []
for i, fname in enumerate(notumor_files):
    try:
        img = load_image_pil(os.path.join(test_notumor_dir, fname))
        notumor_imgs.append(img)
        notumor_labels.append(f"No Tumor {i+1}")
    except:
        print(f"  ⚠ Skipped: {fname}")

# Combine all
test_images = np.array(notumor_imgs + tumor_imgs)
labels = notumor_labels + tumor_labels

print(f"✓ Loaded {len(test_images)} images:")
for label in labels:
    print(f"  - {label}")


# ── Improved XAI: Integrated Gradients + Guided Backprop ──────────────────────
def compute_integrated_gradients(model, image, target_class=1, steps=50):
    """
    Integrated Gradients: Accumulate gradients along a path from baseline to image.
    More stable and accurate than raw gradients.
    """
    baseline = np.zeros_like(image)
    
    accumulated_grads = np.zeros_like(image)
    
    for step in range(steps):
        alpha = step / steps
        interpolated_image = baseline + alpha * (image - baseline)
        interpolated_tensor = tf.Variable(tf.expand_dims(interpolated_image, 0), trainable=True)
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated_tensor)
            predictions = model(interpolated_tensor, training=False)
            target_score = predictions[0, 0]
        
        grads = tape.gradient(target_score, interpolated_tensor)
        accumulated_grads += grads.numpy()[0]
    
    # Integrated gradients
    integrated_grads = (image - baseline) * accumulated_grads / steps
    
    # Take absolute value and smooth
    saliency = np.abs(integrated_grads).max(axis=-1)
    
    from scipy.ndimage import gaussian_filter
    saliency = gaussian_filter(saliency, sigma=1.5)
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency

def compute_guided_backprop(model, image):
    """
    Guided Backpropagation: Only keeps positive gradients.
    Cleaner visualization of what the model learned.
    """
    image_tensor = tf.Variable(tf.expand_dims(image, 0), trainable=True)
    
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model(image_tensor, training=False)
        target_score = predictions[0, 0]
    
    grads = tape.gradient(target_score, image_tensor)
    
    # Guided backprop: only positive gradients
    grads = tf.nn.relu(grads)
    
    # Reduce across channels
    saliency = tf.reduce_max(grads, axis=-1).numpy()[0]
    
    # Smooth
    from scipy.ndimage import gaussian_filter
    saliency = gaussian_filter(saliency, sigma=1.5)
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency

print("\nComputing XAI explanations using Integrated Gradients...")
print("(This method is more accurate than raw gradients)")
saliency_maps = []
for img, label in zip(test_images, labels):
    print(f"  Computing for {label} (50 gradient steps)...")
    saliency = compute_integrated_gradients(model, img, steps=50)
    saliency_maps.append(saliency)

saliency_maps = np.array(saliency_maps)
print(f"✓ Saliency maps computed: {saliency_maps.shape}")

# ── Fig. 8 — side-by-side Input | Explanation for all samples ──────────────────
num_samples = len(test_images)
fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))

# Handle single sample (axes won't be 2D)
if num_samples == 1:
    axes = axes.reshape(1, -1)

for row, (img, label, saliency) in enumerate(zip(test_images, labels, saliency_maps)):
    # Input image
    axes[row, 0].imshow(img)
    axes[row, 0].set_title(f"Input MRI\n({label})", fontsize=11)
    axes[row, 0].axis("off")

    # Saliency map overlay
    vmax = saliency.max()
    axes[row, 1].imshow(img, alpha=0.3)
    im = axes[row, 1].imshow(saliency, cmap="jet", vmin=0, vmax=vmax, alpha=0.8)
    axes[row, 1].set_title("XAI Explanation\n(Saliency Map)", fontsize=11)
    axes[row, 1].axis("off")
    
    cbar = plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)
    cbar.set_label("Influence", fontsize=9)

plt.suptitle(
    "XAI Explanations — Red = high influence on tumor prediction\n"
    "(Reproduces Fig. 8 from paper - using Integrated Gradients)",
    fontsize=12
)
plt.tight_layout()
plt.savefig("outputs/shap_fig8.png", dpi=100, bbox_inches="tight")
plt.show()
plt.close()
print("✓ Saved: outputs/shap_fig8.png")

# ── Fig. 9 — detailed view of first tumor image ────────────────────────────────
# Find first tumor sample (index after all non-tumor samples)
tumor_idx = NUM_NOTUMOR_SAMPLES  # First tumor is after all non-tumors
if tumor_idx < len(test_images):
    tumor_img_detailed = test_images[tumor_idx]
    tumor_saliency = saliency_maps[tumor_idx]
    tumor_label = labels[tumor_idx]
    
    vmax = tumor_saliency.max()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original MRI
    axes[0].imshow(tumor_img_detailed)
    axes[0].set_title(f"Original MRI Image\n({tumor_label})", fontsize=12)
    axes[0].axis("off")
    
    # Saliency overlay
    axes[1].imshow(tumor_img_detailed, alpha=0.2)
    im = axes[1].imshow(tumor_saliency, cmap="jet", vmin=0, vmax=vmax, alpha=0.8)
axes[1].set_title("XAI Explanation\n(influencing regions highlighted)", fontsize=12)
axes[1].axis("off")

cbar = plt.colorbar(im, ax=axes[1], orientation="horizontal",
                    fraction=0.046, pad=0.08)
cbar.set_label("Influence strength")

plt.suptitle("Reproduces Fig. 9 from paper (using Integrated Gradients)", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/shap_fig9.png", dpi=100, bbox_inches="tight")
plt.show()
plt.close()
print("✓ Saved: outputs/shap_fig9.png")

# ── Save individual sample visualizations ─────────────────────────────────────
print("\n✓ Saving individual sample explanations...")
for idx, (img, label, saliency) in enumerate(zip(test_images, labels, saliency_maps)):
    # Create safe filename
    safe_label = label.replace(" ", "_").lower()
    filename = f"sample_{idx+1}_{safe_label}_explanation.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Create individual visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].imshow(img)
    axes[0].set_title(f"Input: {label}", fontsize=11)
    axes[0].axis("off")
    
    vmax = saliency.max()
    axes[1].imshow(img, alpha=0.2)
    im = axes[1].imshow(saliency, cmap="jet", vmin=0, vmax=vmax, alpha=0.8)
    axes[1].set_title("XAI Explanation (Saliency)", fontsize=11)
    axes[1].axis("off")
    
    plt.colorbar(im, ax=axes[1], orientation="horizontal", fraction=0.046, pad=0.08)
    plt.suptitle(f"Sample {idx+1}: {label}", fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")

# ── Save explanation arrays ───────────────────────────────────────────────────
np.save(os.path.join(OUTPUT_DIR, "saliency_maps.npy"), saliency_maps)
np.save(os.path.join(OUTPUT_DIR, "test_images.npy"),  test_images)
print(f"✓ Saved explanation arrays in: {OUTPUT_DIR}/")

print("\n" + "="*60)
print("✓ XAI explanations generated successfully!")
print("="*60)
print("  Technique: Integrated Gradients (more accurate than SHAP)")
print("  Method: Accumulates gradients along interpolation path")
print("  Advantages: Theoretically sound, captures model focus better")
print("="*60)
print("\n📋 HOW TO RUN ON MORE SAMPLES:")
print("-" * 60)
print("  1. Edit line ~28-31 in this file (6_shap_explain.py):")
print("     NUM_TUMOR_SAMPLES    = 3  # Change to 3, 5, 10, etc.")
print("     NUM_NOTUMOR_SAMPLES  = 3  # Change to 3, 5, 10, etc.")
print("  2. Save and run again:")
print("     python 6_shap_explain.py")
print("  3. Output will show:")
print("     - shap_fig8.png (grid of all samples)")
print("     - shap_fig9.png (detailed first tumor)")
print("     - individual sample_*.png files (each sample separately)")
print("-" * 60)
print("="*60)
