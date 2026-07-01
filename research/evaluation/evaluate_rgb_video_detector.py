import os
import json
import csv
import numpy as np

from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

import torchvision.transforms as transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import matplotlib.pyplot as plt

import timm

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = r"C:\Users\hanan\Desktop\datasets"

OUTPUTS_DIR = rf"{BASE_DIR}\outputs"

DATA_ROOT = rf"{BASE_DIR}\processed\video_face_sequences"

MODEL_PATH = rf"{OUTPUTS_DIR}\models\phase4_rgb_only_clean_best.pth"

RESULTS_DIR = rf"{OUTPUTS_DIR}\results\rgb_video_detector"

PLOTS_DIR = rf"{OUTPUTS_DIR}\plots\rgb_video_detector"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 224

SEQ_LEN = 16

BATCH_SIZE = 4

NUM_WORKERS = 0

print(f"\n🚀 Device: {DEVICE}")

# =========================================================
# CREATE OUTPUT DIRS
# =========================================================

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(PLOTS_DIR, exist_ok=True)

# =========================================================
# TRANSFORMS
# =========================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

# =========================================================
# DATASET
# =========================================================

VALID_EXTS = {
    ".jpg",
    ".jpeg",
    ".png"
}

class VideoDataset(Dataset):

    def __init__(self, root, transform=None):

        self.samples = []

        self.transform = transform

        for cls_name, label in [
            ("real", 0),
            ("fake", 1)
        ]:

            cls_dir = os.path.join(root, cls_name)

            for vid in sorted(os.listdir(cls_dir)):

                vid_dir = os.path.join(cls_dir, vid)

                if os.path.isdir(vid_dir):

                    self.samples.append(
                        (vid_dir, label)
                    )

    def _list_frames(self, folder):

        return sorted([
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        ])

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        folder, label = self.samples[idx]

        frames = self._list_frames(folder)

        n_frames = len(frames)

        seq = min(SEQ_LEN, n_frames)

        idxs = np.linspace(
            0,
            n_frames - 1,
            seq
        ).astype(int)

        imgs = []

        for i in idxs:

            img = Image.open(
                os.path.join(folder, frames[i])
            ).convert("RGB")

            img = self.transform(img)

            imgs.append(img)

        while len(imgs) < SEQ_LEN:

            imgs.append(imgs[-1].clone())

        return (
            torch.stack(imgs),
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )

# =========================================================
# DATALOADER
# =========================================================

test_ds = VideoDataset(
    os.path.join(DATA_ROOT, "test"),
    transform
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda")
)

print(f"Test samples: {len(test_ds)}")

# =========================================================
# ATTENTION
# =========================================================

class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):

        weights = torch.softmax(
            self.attn(x),
            dim=1
        )

        return (x * weights).sum(dim=1)

# =========================================================
# MODEL
# =========================================================

class RGBDetector(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = timm.create_model(
            "xception",
            pretrained=False,
            num_classes=0
        )

        with torch.no_grad():

            probe = backbone(
                torch.zeros(
                    1,
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            )

        feat_dim = probe.shape[-1]

        self.backbone = backbone

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.attn = TemporalAttention(256)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),

            nn.Linear(512, 256),

            nn.GELU(),

            nn.Dropout(0.3),

            nn.Linear(256, 1),
        )

    def forward(self, x):

        B, T, C, H, W = x.shape

        feat = self.backbone(
            x.view(B * T, C, H, W)
        )

        feat = self.proj(feat)

        feat = feat.view(B, T, -1)

        out, _ = self.lstm(feat)

        feat = self.attn(out)

        return self.classifier(feat).squeeze(1)

# =========================================================
# LOAD MODEL
# =========================================================

model = RGBDetector().to(DEVICE)

ckpt = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    ckpt["model"]
)

TEMPERATURE = ckpt.get(
    "temperature",
    1.0
)

model.eval()

print(f"✅ Loaded model: {MODEL_PATH}")

# =========================================================
# EVALUATION
# =========================================================

all_labels = []
all_probs = []
all_preds = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(DEVICE)

        with autocast(device_type=DEVICE):

            logits = model(x)

        probs = torch.sigmoid(
            logits / TEMPERATURE
        ).cpu().numpy()

        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs.tolist())

        all_preds.extend(preds.tolist())

        all_labels.extend(
            y.numpy().tolist()
        )

all_labels = np.array(all_labels)

all_probs = np.array(all_probs)

all_preds = np.array(all_preds)

# =========================================================
# METRICS
# =========================================================

acc = accuracy_score(
    all_labels,
    all_preds
)

prec = precision_score(
    all_labels,
    all_preds
)

rec = recall_score(
    all_labels,
    all_preds
)

f1 = f1_score(
    all_labels,
    all_preds
)

auc = roc_auc_score(
    all_labels,
    all_probs
)

cm = confusion_matrix(
    all_labels,
    all_preds
)

# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 50)

print("🔥 RGB VIDEO DETECTOR RESULTS")

print("=" * 50)

print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")

print(f"Precision : {prec:.4f}")

print(f"Recall    : {rec:.4f}")

print(f"F1 Score  : {f1:.4f}")

print(f"ROC-AUC   : {auc:.4f}")

print("\nConfusion Matrix:")

print(cm)

print("=" * 50)

# =========================================================
# SAVE METRICS
# =========================================================

metrics = {
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "roc_auc": float(auc)
}

with open(
    os.path.join(
        RESULTS_DIR,
        "metrics.json"
    ),
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

# =========================================================
# SAVE PREDICTIONS
# =========================================================

with open(
    os.path.join(
        RESULTS_DIR,
        "predictions.csv"
    ),
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

        writer.writerow([
            t,
            p,
            pr
        ])

# =========================================================
# SAVE CONFUSION MATRIX
# =========================================================

plt.figure(figsize=(5, 5))

plt.imshow(cm)

plt.title("RGB Video Detector Confusion Matrix")

plt.colorbar()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "confusion_matrix.png"
    )
)

plt.close()

print(f"\n✅ Results saved in: {RESULTS_DIR}")

print(f"✅ Plots saved in: {PLOTS_DIR}")