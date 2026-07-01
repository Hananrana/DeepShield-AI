import torch
import torch.nn as nn

from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

import cv2
import numpy as np

import json
import csv
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# =========================
# CONFIG
# =========================

BASE_DIR = r"C:\Users\hanan\Desktop\datasets"

OUTPUTS_DIR = r"C:\Users\hanan\Desktop\datasets\outputs"

RGB_DIR = rf"{BASE_DIR}\processed\faces_haar"

FFT_DIR = rf"{BASE_DIR}\processed\fft_mtcnn"

CHECKPOINT_PATH = rf"{OUTPUTS_DIR}\models\fusion_rgb_fft_best.pth"

RESULTS_DIR = Path(
    rf"{OUTPUTS_DIR}\results\fusion_v1"
)

PLOTS_DIR = Path(
    rf"{OUTPUTS_DIR}\plots\fusion_v1"
)

BATCH_SIZE = 32

IMG_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Device: {DEVICE}")

# =========================
# CREATE OUTPUT DIRS
# =========================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# DATASET
# =========================

class DualDataset(Dataset):

    def __init__(self, rgb_root, fft_root, split="test"):

        self.samples = []

        rgb_root = Path(rgb_root) / split

        fft_root = Path(fft_root) / split

        for label, cls in enumerate(["real", "fake"]):

            rgb_dir = rgb_root / cls

            fft_dir = fft_root / cls

            for img_path in rgb_dir.glob("*"):

                # ✅ CORRECT FIX
                fft_path = fft_dir / (img_path.stem + ".png")

                if fft_path.exists():

                    self.samples.append(
                        (
                            img_path,
                            fft_path,
                            label
                        )
                    )

        print(f"Loaded {len(self.samples)} paired samples")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        rgb_path, fft_path, label = self.samples[idx]

        rgb = cv2.imread(str(rgb_path))

        rgb = cv2.cvtColor(
            rgb,
            cv2.COLOR_BGR2RGB
        )

        rgb = cv2.resize(
            rgb,
            (IMG_SIZE, IMG_SIZE)
        )

        fft = cv2.imread(str(fft_path))

        fft = cv2.cvtColor(
            fft,
            cv2.COLOR_BGR2RGB
        )

        fft = cv2.resize(
            fft,
            (IMG_SIZE, IMG_SIZE)
        )

        rgb = self.transform(rgb)

        fft = self.transform(fft)

        return (
            rgb,
            fft,
            torch.tensor(label, dtype=torch.float32)
        )

# =========================
# MODEL
# =========================

class FusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        rgb_backbone = models.efficientnet_b0(
            weights=None
        )

        rgb_dim = rgb_backbone.classifier[1].in_features

        rgb_backbone.classifier = nn.Identity()

        self.rgb_encoder = rgb_backbone

        fft_backbone = models.efficientnet_b0(
            weights=None
        )

        fft_dim = fft_backbone.classifier[1].in_features

        fft_backbone.classifier = nn.Identity()

        self.fft_encoder = fft_backbone

        self.fusion_head = nn.Sequential(
            nn.Linear(rgb_dim + fft_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, rgb, fft):

        rgb_feat = self.rgb_encoder(rgb)

        fft_feat = self.fft_encoder(fft)

        fused = torch.cat(
            [rgb_feat, fft_feat],
            dim=1
        )

        return self.fusion_head(fused)

# =========================
# LOAD MODEL
# =========================

model = FusionModel().to(DEVICE)

ckpt = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    ckpt["model_state"]
)

threshold = ckpt.get(
    "threshold",
    0.5
)

model.eval()

print("✅ Model loaded:", CHECKPOINT_PATH)

print(f"✅ Threshold: {threshold:.3f}")

# =========================
# DATA LOADER
# =========================

test_ds = DualDataset(
    RGB_DIR,
    FFT_DIR,
    split="test"
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# EVALUATION
# =========================

y_true = []

y_pred = []

y_probs = []

with torch.no_grad():

    for rgb, fft, labels in test_loader:

        rgb = rgb.to(DEVICE)

        fft = fft.to(DEVICE)

        outputs = model(rgb, fft)

        probs = torch.sigmoid(
            outputs
        ).squeeze()

        preds = (probs > threshold).float()

        y_true.extend(labels.numpy())

        y_pred.extend(preds.cpu().numpy())

        y_probs.extend(probs.cpu().numpy())

# =========================
# METRICS
# =========================

acc = accuracy_score(
    y_true,
    y_pred
)

prec = precision_score(
    y_true,
    y_pred
)

rec = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

auc = roc_auc_score(
    y_true,
    y_probs
)

cm = confusion_matrix(
    y_true,
    y_pred
)

# =========================
# PRINT RESULTS
# =========================

print("\n==============================")

print("FUSION V1 RESULTS")

print("==============================")

print(f"Accuracy  : {acc:.4f}")

print(f"Precision : {prec:.4f}")

print(f"Recall    : {rec:.4f}")

print(f"F1 Score  : {f1:.4f}")

print(f"AUC-ROC   : {auc:.4f}")

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
    "auc": float(auc),
    "threshold": float(threshold)
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
        y_true,
        y_probs,
        y_pred
    ):
        writer.writerow([t, p, pr])

# =========================
# SAVE CONFUSION MATRIX
# =========================

plt.figure(figsize=(5,5))

plt.imshow(cm)

plt.title("Fusion V1 Confusion Matrix")

plt.colorbar()

plt.savefig(
    PLOTS_DIR / "confusion_matrix.png"
)

plt.close()

print("\n✅ Results saved in:")

print(RESULTS_DIR)

print("\n✅ Plots saved in:")

print(PLOTS_DIR)

print("\n✅ Evaluation complete.")