"""
Step 7: Model Comparison — Literature Review Table
Reproduces Table 3 from the paper:
  "Utilizing Customized CNN for Brain Tumor Prediction with Explainable AI"
  Heliyon/Elsevier 2024

NOTE: The paper does NOT train these comparison models itself.
It compares results published in other papers on the same BR35H dataset.
We reproduce this as a formatted table + bar chart, then add our own
binary model result at the bottom for comparison.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

os.makedirs("outputs", exist_ok=True)

# ── Table 3 from paper — results reported in cited literature ─────────────────
literature_results = [
    {"Authors": "Remzan et al. [41]",   "Year": 2024, "Method": "SVM",            "XAI": "No",               "Accuracy": 98.67},
    {"Authors": "Albalawi et al. [42]", "Year": 2024, "Method": "Modified VGG16", "XAI": "No",               "Accuracy": 98.00},
    {"Authors": "Haque et al. [13]",    "Year": 2024, "Method": "NeuroNet19",      "XAI": "LIME",             "Accuracy": 99.30},
    {"Authors": "Islam et al. [43]",    "Year": 2024, "Method": "CNN",             "XAI": "No",               "Accuracy": 98.82},
    {"Authors": "Khushi et al. [44]",   "Year": 2024, "Method": "AlexNet",         "XAI": "No",               "Accuracy": 98.79},
    {"Authors": "Ahmed et al. [45]",    "Year": 2023, "Method": "VGG16",           "XAI": "LRP",              "Accuracy": 97.33},
    {"Authors": "Islam et al. [46]",    "Year": 2023, "Method": "CNN",             "XAI": "No",               "Accuracy": 98.50},
    {"Authors": "Garg et al. [47]",     "Year": 2023, "Method": "CNN",             "XAI": "No",               "Accuracy": 98.66},
    {"Authors": "Gomez et al. [48]",    "Year": 2023, "Method": "InceptionV3",     "XAI": "No",               "Accuracy": 97.12},
    {"Authors": "Shah et al. [49]",     "Year": 2023, "Method": "DenseNet121",     "XAI": "No",               "Accuracy": 97.86},
    {"Authors": "Cinar et al. [50]",    "Year": 2022, "Method": "ResNet101",       "XAI": "No",               "Accuracy": 98.60},
    {"Authors": "Naseer et al. [51]",   "Year": 2021, "Method": "CNN",             "XAI": "No",               "Accuracy": 98.80},
    {"Authors": "Paper (Binary CNN)",   "Year": 2024, "Method": "Custom CNN",      "XAI": "SHAP+LIME+GradCAM","Accuracy": 100.00},
]

df_lit = pd.DataFrame(literature_results)

# ── Add our binary model result ───────────────────────────────────────────────
try:
    model = load_model("models/brain_tumor_cnn.h5")
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        "data/Testing", target_size=(150, 150),
        batch_size=8, class_mode="binary", shuffle=False
    )
    y_prob = model.predict(test_gen, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = test_gen.classes
    rpt    = classification_report(y_true, y_pred, target_names=["no", "yes"],
                                   output_dict=True)
    our_acc = round(model.evaluate(test_gen, verbose=0)[1] * 100, 2)
    our_f1  = round(rpt["weighted avg"]["f1-score"] * 100, 2)
    our_row = {
        "Authors": "Our Model (Binary CNN)",
        "Year": 2024,
        "Method": "Custom CNN (Binary)",
        "XAI": "SHAP+LIME+GradCAM",
        "Accuracy": our_acc
    }
    df_lit = pd.concat([df_lit, pd.DataFrame([our_row])], ignore_index=True)
    print(f"Our model accuracy: {our_acc}%  |  F1: {our_f1}%")
except Exception as e:
    print(f"Could not load model yet: {e}. Run 2_train_cnn.py first.")

# ── Print table ───────────────────────────────────────────────────────────────
print("\n── Literature Comparison Table (Table 3 from paper) ──")
print(df_lit.to_string(index=False))
df_lit.to_csv("outputs/literature_comparison.csv", index=False)

# ── Bar chart ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))

colors = []
for _, row in df_lit.iterrows():
    if "Our Model" in row["Authors"]:
        colors.append("#9b59b6")          # purple = our extension
    elif "Paper" in row["Authors"]:
        colors.append("#e74c3c")          # red = paper's proposed model
    elif row["XAI"] != "No":
        colors.append("#f39c12")          # orange = has XAI
    else:
        colors.append("#3498db")          # blue = no XAI

bars = ax.barh(df_lit["Authors"], df_lit["Accuracy"], color=colors, edgecolor="white", height=0.7)

# Annotate values
for bar, val in zip(bars, df_lit["Accuracy"]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val}%", va="center", fontsize=9, fontweight="bold")

# Dynamically set x-axis to include all values
min_acc = df_lit["Accuracy"].min()
max_acc = df_lit["Accuracy"].max()
ax.set_xlim([max(0, min_acc - 5), min(100, max_acc + 5)])

ax.set_xlabel("Accuracy (%)", fontsize=11)
ax.set_title("Model Comparison: Literature Review + Our Model (Table 3 from Paper)", fontsize=13, fontweight="bold")
ax.grid(axis='x', alpha=0.3, linestyle='--')

legend_patches = [
    mpatches.Patch(color="#3498db", label="No XAI"),
    mpatches.Patch(color="#f39c12", label="With XAI"),
    mpatches.Patch(color="#e74c3c", label="Paper's Proposed Model (100%)"),
    mpatches.Patch(color="#9b59b6", label="Our Binary CNN (Current)"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=10, framealpha=0.95)
plt.tight_layout()
plt.savefig("outputs/literature_comparison.png", dpi=100, bbox_inches="tight")
plt.show()
plt.close()
print("\n✓ Saved: outputs/literature_comparison.png")
print("✓ Saved: outputs/literature_comparison.csv")
