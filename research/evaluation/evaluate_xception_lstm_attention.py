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

# =========================================
# CONFIG
# =========================================

BASE_DIR = r"C:\Users\hanan\Desktop\datasets"

OUTPUTS_DIR = r"C:\Users\hanan\Desktop\datasets\outputs"

DATASET_PATH = rf"{BASE_DIR}\processed\video_face_sequences"

MODEL_PATH = rf"{OUTPUTS_DIR}\models\xception_lstm_best.pth"

RESULTS_DIR = rf"{OUTPUTS_DIR}\results\xception_lstm_attention"

PLOTS_DIR = rf"{OUTPUTS_DIR}\plots\xception_lstm_attention"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224

SEQ_LEN = 16

BATCH_SIZE = 4

print(f"\n🚀 Device: {DEVICE}")

# =========================================
# CREATE OUTPUT DIRS
# =========================================

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(PLOTS_DIR, exist_ok=True)

# =========================================
# TRANSFORMS
# =========================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================================
# DATASET
# =========================================

class VideoDataset(Dataset):

    def __init__(self, root, transform=None):

        self.samples = []

        self.transform = transform

        for label, cls in enumerate(["real", "fake"]):

            class_path = os.path.join(root, cls)

            for vid in sorted(os.listdir(class_path)):

                vid_path = os.path.join(class_path, vid)

                if os.path.isdir(vid_path):

                    self.samples.append(
                        (vid_path, label)
                    )

    def __len__(self):

        return len(self.samples)

    def _load_frames(self, vid_path):

        frames = sorted([
            f for f in os.listdir(vid_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        total = len(frames)

        if total >= SEQ_LEN:

            indices = np.linspace(
                0,
                total - 1,
                SEQ_LEN,
                dtype=int
            )

            frames = [frames[i] for i in indices]

        else:

            while len(frames) < SEQ_LEN:

                frames.append(frames[-1])

        return frames

    def __getitem__(self, idx):

        vid_path, label = self.samples[idx]

        frames = self._load_frames(vid_path)

        images = []

        for f in frames:

            img = Image.open(
                os.path.join(vid_path, f)
            ).convert("RGB")

            img = self.transform(img)

            images.append(img)

        return (
            torch.stack(images),
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )

# =========================================
# DATALOADER
# =========================================

test_dataset = VideoDataset(
    os.path.join(DATASET_PATH, "test"),
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print(f"Test samples: {len(test_dataset)}")

# =========================================
# ATTENTION
# =========================================

class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):

        scores = self.attn(lstm_out)

        weights = torch.softmax(scores, dim=1)

        context = (weights * lstm_out).sum(dim=1)

        return context

# =========================================
# MODEL
# =========================================

class XceptionLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = timm.create_model(
            "xception",
            pretrained=False,
            num_classes=0
        )

        feature_dim = self.cnn.num_features

        self.proj = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        lstm_out_dim = 512

        self.attention = TemporalAttention(
            lstm_out_dim
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(lstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):

        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        feat = self.cnn(x)

        feat = self.proj(feat)

        feat = feat.view(B, T, -1)

        lstm_out, _ = self.lstm(feat)

        context = self.attention(lstm_out)

        out = self.classifier(context)

        return out.squeeze(1)

# =========================================
# LOAD MODEL
# =========================================

model = XceptionLSTM().to(DEVICE)

ckpt = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    ckpt["model_state"]
)

model.eval()

print(f"✅ Loaded model: {MODEL_PATH}")

# =========================================
# EVALUATION
# =========================================

all_labels = []

all_probs = []

all_preds = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(DEVICE)

        with autocast(device_type="cuda"):

            outputs = model(x)

        probs = torch.sigmoid(
            outputs
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

# =========================================
# METRICS
# =========================================

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

# =========================================
# PRINT RESULTS
# =========================================

print("\n" + "=" * 50)

print("🔥 XCEPTION + LSTM + ATTENTION RESULTS")

print("=" * 50)

print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")

print(f"Precision : {prec:.4f}")

print(f"Recall    : {rec:.4f}")

print(f"F1 Score  : {f1:.4f}")

print(f"ROC-AUC   : {auc:.4f}")

print("\nConfusion Matrix:")

print(cm)

print("=" * 50)

# =========================================
# SAVE METRICS
# =========================================

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

# =========================================
# SAVE PREDICTIONS
# =========================================

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
            float(p),
            int(pr)
        ])

# =========================================
# SAVE CONFUSION MATRIX
# =========================================

plt.figure(figsize=(5, 5))

plt.imshow(cm)

plt.title(
    "Xception LSTM Attention Confusion Matrix"
)

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