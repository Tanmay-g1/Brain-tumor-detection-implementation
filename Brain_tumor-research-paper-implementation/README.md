

| Paper spec | Our code |
|---|---|
| Dataset: BR35H, binary tumor/no-tumor | ✅ `class_mode="binary"`, yes/no folders |
| Image size: 150×150 | ✅ `IMG_SIZE = (150, 150)` |
| CNN: Conv 32→64→128→256→256 | ✅ 5 conv blocks |
| Total parameters: 1,511,233 | ✅ Verified by math |
| Dense: 128→64→1 | ✅ Sigmoid output |
| Loss: binary crossentropy | ✅ |
| Optimizer: Adam | ✅ |
| Batch size: 8 | ✅ Algorithm 1 |
| Split: 90% train / 10% val | ✅ `validation_split=0.1` |
| Epochs: 30 | ✅ |
| SHAP: DeepExplainer, red/blue pixel map | ✅ Fig 8 & 9 |
| LIME: superpixels, yellow outlines | ✅ Fig 10 & 11 |
| Grad-CAM: last conv layer heatmap | ✅ Fig 12 |
| Model comparison: literature table only | ✅ Table 3 |
| Cross-dataset validation | ✅ Chakrabarty dataset |

---

## Step-by-Step: Everything You Need To Do

---

### STEP 1 — Install Python packages
Open terminal in your project folder and run:
```bash
cd E:/fortransferee/mlproject7/minor2
pip install -r requirements.txt
```

---

### STEP 2 — Download Dataset 1 (BR35H — the main paper dataset)

1. Go to Kaggle and search: **`BR35H Brain Tumor Detection 2020`** by Ahmed Hamada
2. Download and unzip it. You'll get 3 folders: `yes/`, `no/`, `pred/`
3. Arrange them like this inside your project:

```
data/
  Training/
    yes/    ← paste ALL 1500 images from BR35H yes/ here
    no/     ← paste ALL 1500 images from BR35H no/ here
  Testing/
    yes/    ← paste 30 images from BR35H pred/ here  (any 30)
    no/     ← paste remaining 30 images from pred/ here
```

> The `pred/` folder has 60 mixed images — just split them 30/30 into Testing/yes and Testing/no.

---

### STEP 3 — Run the scripts one by one

**Run in this exact order. Wait for each to finish before running the next.**

```bash
# Verifies your data loaded correctly
# Output: outputs/sample_grid.png
python 1_preprocessing.py
```

```bash
# Trains the CNN for 30 epochs
# Takes: ~2 hrs on CPU, ~10 mins on GPU
# Output: models/brain_tumor_cnn.h5
#         outputs/training_curves.png
python 2_train_cnn.py
```

```bash
# Evaluates accuracy, confusion matrix, ROC curve
# Output: outputs/confusion_matrix.png
#         outputs/performance_metrics.png
#         outputs/roc_auc.png
python 3_evaluate.py
```

```bash
# Generates Grad-CAM heatmaps (fast ~1 min)
# Output: outputs/gradcam_all.png
python 4_gradcam.py
```

```bash
# Generates LIME explanations (slow ~10-15 mins)
# Output: outputs/lime_grid.png
#         outputs/lime_detail.png
python 5_lime_explain.py
```

```bash
# Generates SHAP pixel attribution maps (slow ~10-15 mins)
# Output: outputs/shap_fig8.png
#         outputs/shap_fig9.png
python 6_shap_explain.py
```

```bash
# Creates literature comparison table (Table 3 from paper)
# Output: outputs/literature_comparison.png
python 7_model_comparison.py
```

```bash
# Creates all 3 XAI side-by-side (the paper's key figure)
# Output: outputs/xai_combined_all.png
python 8_xai_combined.py
```

---

### STEP 4 — (Optional but recommended) Cross-dataset validation

1. Go to Kaggle → search: **`Brain MRI Images for Brain Tumor Detection`** by Navoneel Chakrabarty
2. Download and place as:
```
data/
  cross_val/
    yes/    ← tumor images from that dataset
    no/     ← non-tumor images from that dataset
```
3. Then run:
```bash
# Tests generalizability on second dataset
# Paper got: Acc=92%, Prec=94%, Rec=93%, F1=93%
# Output: outputs/cross_val_comparison.png
python 9_cross_dataset_validation.py
```

---

### What you'll have at the end

```
outputs/
  sample_grid.png           ← Step 1
  training_curves.png       ← Step 2  (matches Fig. 7 in paper)
  performance_metrics.png   ← Step 3  (matches Fig. 4 in paper)
  confusion_matrix.png      ← Step 3  (matches Fig. 5 in paper)
  roc_auc.png               ← Step 3
  gradcam_all.png           ← Step 4  (matches Fig. 12 in paper)
  lime_grid.png             ← Step 5  (matches Fig. 10 in paper)
  lime_detail.png           ← Step 5  (matches Fig. 11 in paper)
  shap_fig8.png             ← Step 6  (matches Fig. 8 in paper)
  shap_fig9.png             ← Step 6  (matches Fig. 9 in paper)
  literature_comparison.png ← Step 7  (matches Table 3 in paper)
  xai_combined_all.png      ← Step 8  (paper's core contribution)
  cross_val_comparison.png  ← Step 9  (generalizability proof)
```
