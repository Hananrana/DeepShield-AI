import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
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

MODEL_PATH = rf"{OUTPUTS_DIR}\models\best_cnn.pth"

RESULTS_DIR = Path(
    rf"{OUTPUTS_DIR}\results\cnn_evaluation"
)

PLOTS_DIR = Path(
    rf"{OUTPUTS_DIR}\plots\cnn"
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
# TRANSFORM (same as training eval)
# =========================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================
# DATA
# =========================

test_ds = datasets.ImageFolder(TEST_DIR, transform=transform)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE
)

print("Classes:", test_ds.classes)
print(f"Test samples: {len(test_ds):,}")

# =========================
# MODEL (same as training)
# =========================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN().to(DEVICE)

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

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(DEVICE)

        outputs = model(x)

        probs = torch.sigmoid(outputs).cpu().numpy()

        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs.flatten())
        all_preds.extend(preds.flatten())
        all_labels.extend(y.numpy())

all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
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

print("\n=== CNN TEST RESULTS ===")

print(f"Accuracy : {acc:.4f} ({acc*100:.1f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC-ROC  : {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

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

with open(RESULTS_DIR / "metrics.json", "w") as f:
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

plt.title("CNN Confusion Matrix")

plt.colorbar()

plt.savefig(
    PLOTS_DIR / "confusion_matrix.png"
)

plt.close()

print("\n✅ Results saved in:")
print(RESULTS_DIR)

print("\n✅ Plots saved in:")
print(PLOTS_DIR)