# =========================================================
# PHASE 4 — RGB-ONLY DEEPFAKE DETECTOR
# Xception + BiLSTM + Temporal Attention
# Stable: RTX 3060 · Windows · AMP · PyTorch 2.x
# =========================================================

import io
import os
import random
import sys

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import timm

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable

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

DATA_ROOT = rf"{BASE_DIR}\processed\video_face_sequences"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 224
SEQ_LEN = 16

BATCH_SIZE = 4
ACCUM_STEPS = 4

EPOCHS = 40
CNN_LR = 1e-5
HEAD_LR = 5e-4
PATIENCE = 8

TEMPERATURE = 1.0

SAVE_PATH = rf"{OUTPUTS_DIR}\models\phase4_rgb_only_clean_best.pth"

NUM_WORKERS = 0

# =========================================================
# DEVICE INFO
# =========================================================

print(f"\n🖥️  Using device : {DEVICE}")

if DEVICE == "cuda":

    print(f"   GPU  : {torch.cuda.get_device_name(0)}")

    print(
        f"   VRAM : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

# =========================================================
# AUGMENTATION HELPERS
# =========================================================

def jpeg_compress(
    img_pil: Image.Image,
    quality: int | None = None
) -> Image.Image:

    q = quality if quality is not None else random.randint(50, 90)

    buf = io.BytesIO()

    img_pil.save(
        buf,
        format="JPEG",
        quality=q
    )

    buf.seek(0)

    return Image.open(buf).copy()

def add_gaussian_noise(
    tensor: torch.Tensor,
    std: float = 0.01
) -> torch.Tensor:

    return torch.clamp(
        tensor + std * torch.randn_like(tensor),
        -3.0,
        3.0
    )

# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 16, IMAGE_SIZE + 16)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.15,
        hue=0.04
    ),

    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.15),

    transforms.RandomApply([
        transforms.Lambda(jpeg_compress)
    ], p=0.25),

    transforms.ToTensor(),

    transforms.RandomApply([
        transforms.Lambda(
            lambda t: add_gaussian_noise(t, std=0.01)
        )
    ], p=0.15),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
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

    def __init__(
        self,
        root: str,
        transform=None
    ) -> None:

        self.transform = transform

        self.samples = []

        for cls_name, label in [
            ("real", 0),
            ("fake", 1)
        ]:

            cls_dir = os.path.join(root, cls_name)

            if not os.path.isdir(cls_dir):

                print(
                    f"   ⚠️ Missing class folder: {cls_dir}"
                )

                continue

            for vid in sorted(os.listdir(cls_dir)):

                vid_dir = os.path.join(cls_dir, vid)

                if not os.path.isdir(vid_dir):

                    continue

                frames = self._list_frames(vid_dir)

                if len(frames) == 0:

                    print(
                        f"   ⚠️ Empty folder skipped: {vid_dir}"
                    )

                    continue

                self.samples.append(
                    (vid_dir, label)
                )

        print(
            f"   Loaded {len(self.samples)} video samples"
        )

    @staticmethod
    def _list_frames(folder: str):

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

            if self.transform:

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
# DATALOADERS
# =========================================================

print("\n📂 Building datasets…")

train_ds = VideoDataset(
    os.path.join(DATA_ROOT, "train"),
    train_transform
)

val_ds = VideoDataset(
    os.path.join(DATA_ROOT, "val"),
    val_transform
)

test_ds = VideoDataset(
    os.path.join(DATA_ROOT, "test"),
    val_transform
)

_kw = dict(
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
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
# TEMPORAL ATTENTION
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

        _backbone = timm.create_model(
            "xception",
            pretrained=True,
            num_classes=0
        )

        with torch.no_grad():

            _probe = _backbone(
                torch.zeros(
                    1,
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            )

        feat_dim = _probe.shape[-1]

        del _probe

        self.backbone = _backbone

        for p in self.backbone.parameters():

            p.requires_grad = False

        for p in list(self.backbone.parameters())[-80:]:

            p.requires_grad = True

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
# INIT
# =========================================================

print("\n🏗️  Building model…")

model = RGBDetector().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

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
    },
])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3,
    min_lr=1e-7
)

scaler = GradScaler()

# =========================================================
# CALIBRATION
# =========================================================

def calibrated_sigmoid(
    logits,
    T=1.0
):

    return torch.sigmoid(logits / T)

# =========================================================
# TRAIN EPOCH
# =========================================================

def train_epoch(epoch):

    model.train()

    running_loss = 0.0

    optimizer.zero_grad()

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch:03d}",
        leave=False
    )

    for step, (x, y) in enumerate(pbar):

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        with autocast(device_type=DEVICE):

            logits = model(x)

            loss = criterion(
                logits,
                y
            ) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (
            (step + 1) % ACCUM_STEPS == 0
            or
            (step + 1) == len(train_loader)
        ):

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(optimizer)

            scaler.update()

            optimizer.zero_grad()

        running_loss += (
            loss.item() * ACCUM_STEPS
        )

        pbar.set_postfix(
            loss=f"{loss.item() * ACCUM_STEPS:.4f}"
        )

    return running_loss / len(train_loader)

# =========================================================
# EVALUATE
# =========================================================

def evaluate(
    loader,
    T=1.0,
    threshold=0.5
):

    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    all_logits = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)

            with autocast(device_type=DEVICE):

                logits = model(x)

            prob = calibrated_sigmoid(
                logits,
                T
            ).cpu().numpy()

            raw = logits.cpu().numpy()

            all_logits.extend(raw.tolist())

            all_preds.extend(
                (prob > threshold).astype(int).tolist()
            )

            all_probs.extend(prob.tolist())

            all_labels.extend(
                y.cpu().numpy().tolist()
            )

    probs_arr = np.array(all_probs)

    logits_arr = np.array(all_logits)

    print(
        f"   [Prob dist ] "
        f"mean={probs_arr.mean():.3f} "
        f"std={probs_arr.std():.3f}"
    )

    print(
        f"   [Raw logits] "
        f"mean={logits_arr.mean():.3f} "
        f"std={logits_arr.std():.3f}"
    )

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    unique_classes = np.unique(all_labels)

    if len(unique_classes) < 2:

        print(
            "   ⚠️ Only one class present"
        )

        auc = 0.5

    else:

        auc = roc_auc_score(
            all_labels,
            all_probs
        )

    return acc, f1, auc

# =========================================================
# TRAINING LOOP
# =========================================================

print("\n🚀 CLEAN RGB-ONLY TRAINING\n")

best_auc = 0.0

patience_counter = 0

for epoch in range(1, EPOCHS + 1):

    train_loss = train_epoch(epoch)

    val_acc, val_f1, val_auc = evaluate(
        val_loader,
        T=TEMPERATURE
    )

    scheduler.step(val_auc)

    cnn_lr = optimizer.param_groups[0]["lr"]

    head_lr = optimizer.param_groups[1]["lr"]

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"F1: {val_f1:.4f} | "
        f"AUC: {val_auc:.4f} | "
        f"LR: {cnn_lr:.2e}/{head_lr:.2e}"
    )

    if val_auc > best_auc:

        best_auc = val_auc

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_auc": best_auc,
                "temperature": TEMPERATURE,
            },
            SAVE_PATH,
        )

        patience_counter = 0

        print(
            f"  ✅ Best model saved "
            f"(val_auc={best_auc:.4f})"
        )

    else:

        patience_counter += 1

        print(
            f"  ⏳ No improvement "
            f"({patience_counter}/{PATIENCE})"
        )

    if patience_counter >= PATIENCE:

        print(
            f"\n⏹️ Early stopping "
            f"(best AUC={best_auc:.4f})"
        )

        break

# =========================================================
# FINAL TEST
# =========================================================

print("\n📦 Loading best model…")

ckpt = torch.load(
    SAVE_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    ckpt["model"]
)

TEMPERATURE = ckpt.get(
    "temperature",
    TEMPERATURE
)

print(
    f"   Loaded epoch {ckpt['epoch']} "
    f"| val_auc={ckpt['val_auc']:.4f}"
)

try:

    print("\n📊 FINAL TEST EVALUATION")

    test_acc, test_f1, test_auc = evaluate(
        test_loader,
        T=TEMPERATURE
    )

    print("\n" + "=" * 50)

    print("🔥 FINAL RESULTS")

    print("=" * 50)

    print(f"  Accuracy : {test_acc:.4f}")

    print(f"  F1 Score : {test_f1:.4f}")

    print(f"  ROC-AUC  : {test_auc:.4f}")

    print("=" * 50)

except Exception as exc:

    print(
        f"\n❌ Final evaluation failed: {exc}",
        file=sys.stderr
    )

    raise

print(f"\n💾 Saved : {SAVE_PATH}")