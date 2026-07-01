import torch
import torch.nn as nn

from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
from PIL import Image

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

RGB_ROOT = rf"{BASE_DIR}\processed\faces_haar"

FFT_ROOT = rf"{BASE_DIR}\processed\fft_mtcnn"

MODEL_PATH = rf"{OUTPUTS_DIR}\models\fusion_rgb_fft_best_v2.pth"

RESULTS_DIR = Path(
    rf"{OUTPUTS_DIR}\results\fusion_v2"
)

PLOTS_DIR = Path(
    rf"{OUTPUTS_DIR}\plots\fusion_v2"
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

rgb_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

fft_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.0,0.0,0.0],
        [1.0,1.0,1.0]
    )
])

# =========================
# DATASET
# =========================

class DualDataset(Dataset):

    def __init__(self, rgb_root, fft_root, split="test"):

        self.samples = []

        valid_ext = (
            ".jpg",
            ".jpeg",
            ".png"
        )

        for label_str, label_int in [
            ("real",0),
            ("fake",1)
        ]:

            rgb_dir = Path(rgb_root) / split / label_str

            fft_dir = Path(fft_root) / split / label_str

            for img in rgb_dir.iterdir():

                if not img.name.lower().endswith(valid_ext):
                    continue

                # ✅ FIXED MATCHING
                fft_path = fft_dir / (img.stem + ".png")

                if fft_path.exists():

                    self.samples.append(
                        (
                            img,
                            fft_path,
                            label_int
                        )
                    )

        print(f"✅ Loaded {len(self.samples)} paired samples")

        if len(self.samples) == 0:

            raise RuntimeError(
                "❌ No paired samples found.\n"
                "Check:\n"
                "1. RGB path\n"
                "2. FFT path\n"
                "3. filename matching (.jpg vs .png)"
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        rgb_path, fft_path, label = self.samples[idx]

        rgb = rgb_transform(
            Image.open(rgb_path).convert("RGB")
        )

        fft = fft_transform(
            Image.open(fft_path).convert("RGB")
        )

        return (
            rgb,
            fft,
            torch.tensor(label, dtype=torch.float32)
        )

# =========================
# GATED FUSION HEAD
# =========================

class GatedFusionHead(nn.Module):

    def __init__(self, dim=1280):

        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, rgb_feat, fft_feat):

        fused = torch.cat(
            [rgb_feat, fft_feat],
            dim=1
        )

        g = self.gate(fused)

        rgb_w = rgb_feat * g

        fft_w = fft_feat * (1 - g)

        out = torch.cat(
            [rgb_w, fft_w],
            dim=1
        )

        return self.classifier(out)

# =========================
# MODEL
# =========================

class FusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        rgb_backbone = models.efficientnet_b0(
            weights=None
        )

        dim = rgb_backbone.classifier[1].in_features

        rgb_backbone.classifier = nn.Identity()

        self.rgb_encoder = rgb_backbone

        fft_backbone = models.efficientnet_b0(
            weights=None
        )

        fft_backbone.classifier = nn.Identity()

        self.fft_encoder = fft_backbone

        self.fusion_head = GatedFusionHead(dim)

    def forward(self, rgb, fft):

        rgb_feat = self.rgb_encoder(rgb)

        fft_feat = self.fft_encoder(fft)

        return self.fusion_head(
            rgb_feat,
            fft_feat
        )

# =========================
# LOAD MODEL
# =========================

model = FusionModel().to(DEVICE)

ckpt = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    ckpt["model_state"]
)

threshold = ckpt.get(
    "threshold",
    0.5
)

model.eval()

print("✅ Model loaded")

print("Threshold:", threshold)

# =========================
# DATA
# =========================

test_ds = DualDataset(
    RGB_ROOT,
    FFT_ROOT,
    "test"
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE
)

# =========================
# EVALUATION
# =========================

y_true = []

y_probs = []

with torch.no_grad():

    for rgb, fft, labels in test_loader:

        rgb = rgb.to(DEVICE)

        fft = fft.to(DEVICE)

        logits = model(rgb, fft)

        probs = torch.sigmoid(
            logits
        ).cpu().numpy()

        y_probs.extend(probs.flatten())

        y_true.extend(labels.numpy())

y_probs = np.array(y_probs)

y_true = np.array(y_true)

y_pred = (y_probs > threshold).astype(int)

# =========================
# METRICS
# =========================

acc = accuracy_score(
    y_true,
    y_pred
)

prec = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

rec = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

try:

    auc = roc_auc_score(
        y_true,
        y_probs
    )

except:

    auc = 0.0

cm = confusion_matrix(
    y_true,
    y_pred
)

# =========================
# PRINT RESULTS
# =========================

print("\n=== FUSION V2 RESULTS ===")

print(f"Accuracy : {acc:.4f}")

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

plt.title("Fusion V2 Confusion Matrix")

plt.colorbar()

plt.savefig(
    PLOTS_DIR / "confusion_matrix.png"
)

plt.close()

# =========================
# SAVE TXT MATRIX
# =========================

np.savetxt(
    RESULTS_DIR / "confusion_matrix.txt",
    cm,
    fmt="%d"
)

print("\n✅ Results saved in:")

print(RESULTS_DIR)

print("\n✅ Plots saved in:")

print(PLOTS_DIR)