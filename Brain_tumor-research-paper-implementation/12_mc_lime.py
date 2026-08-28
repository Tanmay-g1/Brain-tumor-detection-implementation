"""
Step 12: Multi-Class LIME Explanations
Shows superpixel contributions for each class
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from lime import lime_image
from skimage.segmentation import mark_boundaries

os.makedirs("outputs/multiclass/lime", exist_ok=True)

MODEL_PATH   = "models/brain_tumor_multiclass.h5"
IMG_SIZE     = (224, 224)
CLASS_NAMES  = ['glioma', 'meningioma', 'pituitary', 'notumor']
NUM_CLASSES  = 4
NUM_LIME_SAMPLES = 500

model = load_model(MODEL_PATH)
print("✓ Model loaded: brain_tumor_multiclass.h5")

# ── LIME predictor for multi-class ────────────────────────────────────────────
def predict_fn_lime(images):
    """Return probabilities for all 4 classes."""
    preds = model.predict(images.astype(np.float32) / 255.0, verbose=0)
    return preds  # Shape: (n_samples, 4)

# ── Load sample images ────────────────────────────────────────────────────────
print("\nLoading sample images...")

test_images = {}
lime_explainer = lime_image.LimeImageExplainer(random_state=42)

for class_idx, class_name in enumerate(CLASS_NAMES):
    class_dir = os.path.join("data/multiclass/Testing", class_name)
    
    if os.path.exists(class_dir):
        files = sorted(os.listdir(class_dir))[:1]
        
        for fname in files:
            img_path = os.path.join(class_dir, fname)
            img_pil = keras_image.load_img(img_path, target_size=IMG_SIZE)
            img_01 = keras_image.img_to_array(img_pil) / 255.0
            img_u8 = np.uint8(img_01 * 255)
            
            test_images[class_name] = {
                'img_01': img_01,
                'img_u8': img_u8,
                'path': img_path
            }
            print(f"  ✓ {class_name}: {fname}")

# ── Compute LIME explanations ─────────────────────────────────────────────────
print(f"\nComputing LIME explanations ({NUM_LIME_SAMPLES} samples per image)...")

fig, axes = plt.subplots(NUM_CLASSES, 2, figsize=(10, 4*NUM_CLASSES))

for row, class_name in enumerate(CLASS_NAMES):
    if class_name not in test_images:
        print(f"  ⚠ Skipped {class_name}")
        continue
    
    print(f"  Explaining {class_name}...")
    
    img_data = test_images[class_name]
    img_u8 = img_data['img_u8']
    img_01 = img_data['img_01']
    
    # Get prediction
    pred = model.predict(np.expand_dims(img_01, 0), verbose=0)[0]
    pred_class_idx = np.argmax(pred)
    confidence = pred[pred_class_idx] * 100
    
    # LIME explanation
    lime_exp = lime_image.LimeImageExplainer().explain_instance(
        img_u8, predict_fn_lime, top_labels=NUM_CLASSES,
        hide_color=0, num_samples=NUM_LIME_SAMPLES, random_seed=42
    )
    
    # Get explanation for predicted class
    temp, mask = lime_exp.get_image_and_mask(
        pred_class_idx, positive_only=True, num_features=10, hide_rest=False
    )
    
    # Plot
    # Column 1: Original
    axes[row, 0].imshow(img_01)
    axes[row, 0].set_title(f"{class_name.upper()}", fontsize=11)
    axes[row, 0].axis("off")
    
    # Column 2: LIME explanation
    axes[row, 1].imshow(mark_boundaries(temp / 255.0, mask, color=(1, 1, 0)))
    pred_name = CLASS_NAMES[pred_class_idx]
    axes[row, 1].set_title(f"LIME: {pred_name} ({confidence:.1f}%)", fontsize=11)
    axes[row, 1].axis("off")
    
    # Save individual explanation
    fig_ind, ax_ind = plt.subplots(1, 2, figsize=(10, 4))
    ax_ind[0].imshow(img_01)
    ax_ind[0].set_title(f"Original: {class_name}", fontsize=11)
    ax_ind[0].axis("off")
    
    ax_ind[1].imshow(mark_boundaries(temp / 255.0, mask, color=(1, 1, 0)))
    ax_ind[1].set_title(f"LIME Explanation", fontsize=11)
    ax_ind[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(f"outputs/multiclass/lime/lime_{class_name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig_ind)

plt.suptitle("Multi-Class LIME: Superpixel Contributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/multiclass/lime_all.png", dpi=100, bbox_inches="tight")
plt.close()

print("✓ Saved: outputs/multiclass/lime_all.png")
print("✓ Next: python 13_mc_shap.py\n")
