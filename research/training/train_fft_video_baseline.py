# =========================================================
# PHASE 3 — FFT-ONLY BASELINE
# Xception + BiLSTM + Temporal Attention
# =========================================================

import os
import random
import numpy as np

from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import torchvision.transforms as transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

import timm

from tqdm import tqdm

import torch._dynamo

torch._dynamo.config.suppress_errors = True

# =========================================================
# REPRODUCIBILITY
# =========================================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = False

torch.backends.cudnn.benchmark = True

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = r"C:\Users\hanan\Desktop\datasets"

OUTPUTS_DIR = r"C:\Users\hanan\Desktop\datasets\outputs"

DATASET_ROOT = rf"{BASE_DIR}\processed\video_fft_sequences"

SAVE_PATH = rf"{OUTPUTS_DIR}\models\phase3_fft_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 224

SEQ_LEN = 16

BATCH_SIZE = 4

ACCUM_STEPS = 4

EPOCHS = 40

CNN_LR = 5e-6

HEAD_LR = 5e-4

PATIENCE = 8

# =========================================================
# DEVICE INFO
# =========================================================

print(f"\n🖥️  Using device: {DEVICE}")

if DEVICE == "cuda":

    print(
        f"   GPU  : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"   VRAM : "
        f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB"
    )

# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

# =========================================================
# DATASET
# =========================================================

class FFTVideoDataset(Dataset):

    def __init__(self, root, transform=None):

        self.samples = []

        self.transform = transform

        for cls_name, label in [
            ("real", 0),
            ("fake", 1)
        ]:

            cls_path = os.path.join(root, cls_name)

            if not os.path.isdir(cls_path):

                raise FileNotFoundError(
                    f"Missing: {cls_path}"
                )

            for vid in sorted(os.listdir(cls_path)):

                vid_path = os.path.join(cls_path, vid)

                if os.path.isdir(vid_path):

                    self.samples.append(
                        (vid_path, label)
                    )

        print(
            f"   Loaded {len(self.samples)} samples from {root}"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        vid_path, label = self.samples[idx]

        frames = sorted([
            f for f in os.listdir(vid_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        if not frames:

            raise ValueError(
                f"Empty folder: {vid_path}"
            )

        idxs = np.linspace(
            0,
            len(frames) - 1,
            SEQ_LEN
        ).astype(int)

        selected = [frames[i] for i in idxs]

        imgs = []

        for frame in selected:

            img = Image.open(
                os.path.join(vid_path, frame)
            ).convert("RGB")

            if self.transform:

                img = self.transform(img)

            imgs.append(img)

        return (
            torch.stack(imgs),
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )

# =========================================================
# DATALOADERS
# =========================================================

print("\n📂 Building datasets...")

train_ds = FFTVideoDataset(
    os.path.join(DATASET_ROOT, "train"),
    train_transform
)

val_ds = FFTVideoDataset(
    os.path.join(DATASET_ROOT, "val"),
    val_transform
)

test_ds = FFTVideoDataset(
    os.path.join(DATASET_ROOT, "test"),
    val_transform
)

_kw = dict(
    num_workers=0,
    pin_memory=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    **_kw
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    **_kw
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    **_kw
)

# =========================================================
# CLASS BALANCE
# =========================================================

real_count = sum(
    1 for _, l in train_ds.samples
    if l == 0
)

fake_count = sum(
    1 for _, l in train_ds.samples
    if l == 1
)

pos_weight = torch.tensor(
    [real_count / fake_count],
    device=DEVICE
)

print(
    f"\n⚖️  real: {real_count} | fake: {fake_count}"
)

print(
    f"   pos_weight: {pos_weight.item():.4f}"
)

# =========================================================
# TEMPORAL ATTENTION
# =========================================================

class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
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

class FFTDetector(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            "xception",
            pretrained=True,
            num_classes=0
        )

        for p in self.backbone.parameters():

            p.requires_grad = False

        for p in list(self.backbone.parameters())[-60:]:

            p.requires_grad = True

        self.proj = nn.Sequential(
            nn.Linear(2048, 512),
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

        self.attn = TemporalAttention(256)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):

        B, T, C, H, W = x.shape

        feat = self.backbone(
            x.view(B * T, C, H, W)
        )

        feat = self.proj(feat)

        feat = feat.view(B, T, -1)

        out, _ = self.lstm(feat)

        out = self.attn(out)

        return self.classifier(out).squeeze(1)

# =========================================================
# INIT
# =========================================================

print("\n🏗️  Building FFT model...")

model = FFTDetector().to(DEVICE)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

cnn_params = [
    p for n, p in model.named_parameters()
    if "backbone" in n and p.requires_grad
]

head_params = [
    p for n, p in model.named_parameters()
    if "backbone" not in n and p.requires_grad
]

optimizer = torch.optim.AdamW([
    {
        "params": cnn_params,
        "lr": CNN_LR,
        "weight_decay": 1e-4
    },
    {
        "params": head_params,
        "lr": HEAD_LR,
        "weight_decay": 1e-4
    }
], fused=True)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3,
    min_lr=1e-7
)

scaler = GradScaler(device="cuda")