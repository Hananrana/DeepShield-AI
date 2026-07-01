import json
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = r"C:\Users\hanan\Desktop\datasets"

OUTPUTS_DIR = r"C:\Users\hanan\Desktop\datasets\outputs"

TEST_DIR = rf"{BASE_DIR}\processed\fft_mtcnn\test"

MODEL_PATH = rf"{OUTPUTS_DIR}\models\fft_model_best.pth"

RESULTS_DIR = Path(
    rf"{OUTPUTS_DIR}\results\fft_model"
)

PLOTS_DIR = Path(
    rf"{OUTPUTS_DIR}\plots\fft_model"
)

BATCH_SIZE = 32

IMAGE_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Device: {DEVICE}")

# =============================================================================
# CREATE OUTPUT DIRS
# =============================================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FFT NORMALIZATION
# =============================================================================

_FFT_MEAN = [0.0, 0.0, 0.0]

_FFT_STD  = [1.0, 1.0, 1.0]

# =============================================================================
# TRANSFORMS
# =============================================================================

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_FFT_MEAN, _FFT_STD),
])

# =============================================================================
# DATA
# =============================================================================

test_ds = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Classes:", test_ds.classes)

print(f"Test samples: {len(test_ds):,}")

# =============================================================================
# MODEL
# =============================================================================

model = models.efficientnet_b0()

in_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, 1),
)

model = model.to(DEVICE)

# =============================================================================
# LOAD CHECKPOINT
# =============================================================================

ckpt = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    ckpt["model_state"]
)

threshold = float(
    ckpt["threshold"]
)

model.eval()

print("✅ Model loaded:", MODEL_PATH)

print(f"✅ Threshold: {threshold:.3f}")

# =============================================================================
# EVALUATION
# =============================================================================

all_probs  = []
all_preds  = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        logits = model(images)

        probs = torch.sigmoid(
            logits
        ).cpu().view(-1).numpy()

        preds = (probs > threshold).astype(int)

        all_probs.extend(probs)

        all_preds.extend(preds)

        all_labels.extend(labels.numpy())

all_probs  = np.array(all_probs)

all_preds  = np.array(all_preds)

all_labels = np.array(all_labels)

# =============================================================================
# METRICS
# =============================================================================

acc = accuracy_score(
    all_labels,
    all_preds
)

prec = precision_score(
    all_labels,
    all_preds,
    zero_division=0
)

rec = recall_score(
    all_labels,
    all_preds,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_preds,
    zero_division=0
)

auc = roc_auc_score(
    all_labels,
    all_probs
)

cm = confusion_matrix(
    all_labels,
    all_preds
)

print("\n=== FFT MODEL TEST RESULTS ===")

print(f"Accuracy : {acc:.4f} ({acc*100:.1f}%)")

print(f"Precision: {prec:.4f}")

print(f"Recall   : {rec:.4f}")

print(f"F1 Score : {f1:.4f}")

print(f"AUC-ROC  : {auc:.4f}")

print("\nConfusion Matrix:")

print(cm)

print(f"\nTN: {cm[0][0]} | FP: {cm[0][1]}")

print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")

# =============================================================================
# SAVE METRICS
# =============================================================================

metrics = {
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1": float(f1),
    "auc": float(auc),
    "threshold": float(threshold)
}

with open(
    RESULTS_DIR / "metrics.json",
    "w"
) as f:

    json.dump(metrics, f, indent=4)

# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

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

# =============================================================================
# SAVE CONFUSION MATRIX
# =============================================================================

plt.figure(figsize=(5,5))

plt.imshow(cm)

plt.title("FFT Model Confusion Matrix")

plt.colorbar()

plt.savefig(
    PLOTS_DIR / "confusion_matrix.png"
)

plt.close()

print("\n✅ Results saved in:")

print(RESULTS_DIR)

print("\n✅ Plots saved in:")

print(PLOTS_DIR)