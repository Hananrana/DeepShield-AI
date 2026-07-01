import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

import numpy as np
import json
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================

BASE_DIR = r"C:\Users\hanan\Desktop\datasets"

OUTPUTS_DIR = r"C:\Users\hanan\Desktop\datasets\outputs"

TEST_DIR = rf"{BASE_DIR}\processed\faces_haar\test"

MODEL_PATH = rf"{OUTPUTS_DIR}\models\best_efficientnet_final.pth"

RESULTS_DIR = Path(
    rf"{OUTPUTS_DIR}\results\efficientnet_b0"
)

PLOTS_DIR = Path(
    rf"{OUTPUTS_DIR}\plots\efficientnet_b0"
)

BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Device: {DEVICE}")

# =========================
# CREATE OUTPUT DIRS
# =========================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# TRANSFORMS
# =========================

eval_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# =========================
# DATA
# =========================

test_ds = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_tf
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Classes:", test_ds.classes)

print(f"Test samples: {len(test_ds):,}")

# =========================
# MODEL
# =========================

model = models.efficientnet_b0()

in_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 1)
)

model = model.to(DEVICE)

# =========================
# LOAD MODEL
# =========================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("✅ Model loaded:", MODEL_PATH)

# =========================
# EVALUATION
# =========================

all_preds  = []
all_labels = []
all_probs  = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(DEVICE)

        outputs = model(x)

        probs = torch.sigmoid(outputs).cpu().numpy()

        preds = (probs > 0.6).astype(int)

        all_probs.extend(probs.flatten())

        all_preds.extend(preds.flatten())

        all_labels.extend(y.numpy())

all_probs  = np.array(all_probs)

all_preds  = np.array(all_preds)

all_labels = np.array(all_labels)

# =========================
# METRICS
# =========================

acc  = accuracy_score(all_labels, all_preds)

prec = precision_score(all_labels, all_preds)

rec  = recall_score(all_labels, all_preds)

f1   = f1_score(all_labels, all_preds)

auc  = roc_auc_score(all_labels, all_probs)

cm = confusion_matrix(
    all_labels,
    all_preds
)

print("\n=== EfficientNet TEST RESULTS ===")

print(f"Accuracy : {acc:.4f} ({acc*100:.1f}%)")

print(f"Precision: {prec:.4f}")

print(f"Recall   : {rec:.4f}")

print(f"F1 Score : {f1:.4f}")

print(f"AUC-ROC  : {auc:.4f}")

print("\nConfusion Matrix:")

print(cm)

print(f"\nTN: {cm[0][0]} | FP: {cm[0][1]}")

print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")

# =========================
# SAVE METRICS
# =========================

metrics = {
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1": float(f1),
    "auc": float(auc)
}

with open(
    RESULTS_DIR / "metrics.json",
    "w"
) as f:

    json.dump(metrics, f, indent=4)

# =========================
# SAVE PREDICTIONS
# =========================

with open(
    RESULTS_DIR / "predictions.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "y_true",
        "y_prob",
        "y_pred"
    ])

    for t, p, pr in zip(
        all_labels,
        all_probs,
        all_preds
    ):
        writer.writerow([t, p, pr])

# =========================
# SAVE CONFUSION MATRIX
# =========================

plt.figure(figsize=(5,5))

plt.imshow(cm)

plt.title("EfficientNet Confusion Matrix")

plt.colorbar()

plt.savefig(
    PLOTS_DIR / "confusion_matrix.png"
)

plt.close()

print("\n✅ Results saved in:")

print(RESULTS_DIR)

print("\n✅ Plots saved in:")

print(PLOTS_DIR)