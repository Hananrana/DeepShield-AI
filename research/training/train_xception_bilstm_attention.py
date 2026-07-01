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

# =========================================
# SEED — Full reproducibility
# =========================================

SEED = 42

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)

np.random.seed(SEED)

random.seed(SEED)

torch.backends.cudnn.deterministic = False

torch.backends.cudnn.benchmark = True

# =========================================
# CONFIG
# =========================================

BASE_DIR = r"C:\Users\hanan\Desktop\datasets"

OUTPUTS_DIR = r"C:\Users\hanan\Desktop\datasets\outputs"

DATASET_PATH = rf"{BASE_DIR}\processed\video_face_sequences"

MODEL_SAVE_PATH = rf"{OUTPUTS_DIR}\models\xception_lstm_best.pth"

RESUME_CHECKPOINT = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n🖥️  Using device: {DEVICE}")

if DEVICE == "cuda":

    print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print(
        f"   VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

IMG_SIZE = 224

SEQ_LEN_MODE = "quality"

if SEQ_LEN_MODE == "quality":

    SEQ_LEN = 16

    BATCH_SIZE = 4

    ACCUM_STEPS = 4

else:

    SEQ_LEN = 8

    BATCH_SIZE = 8

    ACCUM_STEPS = 2

EPOCHS = 40

LEARNING_RATE_CNN = 5e-6

LEARNING_RATE_HEAD = 5e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING = 8

LABEL_SMOOTHING = 0.05

# =========================================
# TRANSFORMS
# =========================================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.05),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.05
    ),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
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

    def __init__(self, root, transform=None, seq_len=SEQ_LEN):

        self.samples = []

        self.transform = transform

        self.seq_len = seq_len

        for label, cls in enumerate(["real", "fake"]):

            class_path = os.path.join(root, cls)

            if not os.path.isdir(class_path):

                raise FileNotFoundError(
                    f"Missing class folder: {class_path}"
                )

            for vid in sorted(os.listdir(class_path)):

                vid_path = os.path.join(class_path, vid)

                if os.path.isdir(vid_path):

                    self.samples.append(
                        (vid_path, label)
                    )

        print(
            f"   Loaded {len(self.samples)} samples from {root}"
        )

    def __len__(self):

        return len(self.samples)

    def _load_frames(self, vid_path):

        frames = sorted([
            f for f in os.listdir(vid_path)
            if f.lower().endswith(
                (".jpg", ".png", ".jpeg")
            )
        ])

        if not frames:

            raise ValueError(
                f"Empty folder: {vid_path}"
            )

        total = len(frames)

        if total >= self.seq_len:

            indices = np.linspace(
                0,
                total - 1,
                self.seq_len,
                dtype=int
            )

            frames = [frames[i] for i in indices]

        else:

            while len(frames) < self.seq_len:

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

            if self.transform:

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
# DATALOADERS
# =========================================

print("\n📂 Building datasets...")

train_dataset = VideoDataset(
    os.path.join(DATASET_PATH, "train"),
    transform=train_transform
)

val_dataset = VideoDataset(
    os.path.join(DATASET_PATH, "val"),
    transform=val_transform
)

test_dataset = VideoDataset(
    os.path.join(DATASET_PATH, "test"),
    transform=val_transform
)

_loader_kwargs = dict(
    batch_size=BATCH_SIZE,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False,
)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    **_loader_kwargs
)

val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    **_loader_kwargs
)

test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    **_loader_kwargs
)

# =========================================
# POS WEIGHT
# =========================================

real_count = sum(
    1 for _, l in train_dataset.samples
    if l == 0
)

fake_count = sum(
    1 for _, l in train_dataset.samples
    if l == 1
)

pos_weight = torch.tensor(
    [real_count / fake_count],
    device=DEVICE
)

print(
    f"\n⚖️  Class balance — "
    f"real: {real_count} | fake: {fake_count}"
)

print(
    f"   pos_weight: {pos_weight.item():.4f}"
)

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
            pretrained=True,
            num_classes=0
        )

        feature_dim = self.cnn.num_features

        for p in self.cnn.parameters():

            p.requires_grad = False

        for p in list(self.cnn.parameters())[-60:]:

            p.requires_grad = True

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
# INIT
# =========================================

print("\n🏗️  Building model...")

model = XceptionLSTM().to(DEVICE)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.AdamW([
    {
        "params": model.cnn.parameters(),
        "lr": LEARNING_RATE_CNN,
        "weight_decay": WEIGHT_DECAY
    },
    {
        "params":
            list(model.proj.parameters()) +
            list(model.lstm.parameters()) +
            list(model.attention.parameters()) +
            list(model.classifier.parameters()),
        "lr": LEARNING_RATE_HEAD,
        "weight_decay": WEIGHT_DECAY
    }
], fused=True)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
)

scaler = GradScaler(device="cuda")

# =========================================
# TRAIN
# =========================================

def train_epoch(epoch):

    model.train()

    total_loss = 0.0

    optimizer.zero_grad()

    loop = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        leave=False
    )

    for step, (x, y) in enumerate(loop):

        x = x.to(DEVICE, non_blocking=True)

        y = y.to(DEVICE, non_blocking=True)

        with autocast(device_type="cuda"):

            out = model(x)

            loss = criterion(out, y) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (
            (step + 1) % ACCUM_STEPS == 0
            or
            (step + 1) == len(train_loader)
        ):

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)

            scaler.update()

            optimizer.zero_grad()

        total_loss += loss.item() * ACCUM_STEPS

        loop.set_postfix(
            loss=f"{loss.item() * ACCUM_STEPS:.4f}"
        )

    return total_loss / len(train_loader)

# =========================================
# EVALUATE
# =========================================

def evaluate(loader, split="val"):

    model.eval()

    all_preds = []

    all_probs = []

    all_labels = []

    with torch.no_grad():

        for x, y in tqdm(
            loader,
            desc=f"  {split}",
            leave=False
        ):

            x = x.to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda"):

                out = model(x)

            prob = torch.sigmoid(out).cpu().numpy()

            pred = (prob > 0.5).astype(float)

            all_probs.extend(prob.tolist())

            all_preds.extend(pred.tolist())

            all_labels.extend(y.numpy().tolist())

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            all_labels,
            all_probs
        )

    except:

        auc = float("nan")

    return acc, f1, auc

# =========================================
# SAVE CHECKPOINT
# =========================================

def save_checkpoint(epoch, acc):

    os.makedirs(
        os.path.dirname(MODEL_SAVE_PATH),
        exist_ok=True
    )

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_acc": acc,
    }, MODEL_SAVE_PATH)

# =========================================
# TRAIN LOOP
# =========================================

print(f"\n🚀 XCEPTION + BiLSTM + ATTENTION\n")

best_acc = 0.0

patience = 0

for epoch in range(1, EPOCHS + 1):

    train_loss = train_epoch(epoch)

    val_acc, val_f1, val_auc = evaluate(
        val_loader,
        split="val"
    )

    scheduler.step(val_acc)

    lr_cnn = optimizer.param_groups[0]["lr"]

    lr_head = optimizer.param_groups[1]["lr"]

    print(
        f"Epoch {epoch:03d}/{EPOCHS}"
        f" | Loss: {train_loss:.4f}"
        f" | Val Acc: {val_acc:.4f}"
        f" | F1: {val_f1:.4f}"
        f" | AUC: {val_auc:.4f}"
        f" | LR CNN/Head: "
        f"{lr_cnn:.2e}/{lr_head:.2e}"
    )

    if val_acc > best_acc:

        best_acc = val_acc

        save_checkpoint(epoch, best_acc)

        print(
            f"  ✅ Best model saved "
            f"(val_acc={best_acc:.4f})"
        )

        patience = 0

    else:

        patience += 1

        print(
            f"  ⏳ No improvement "
            f"({patience}/{EARLY_STOPPING})"
        )

    if patience >= EARLY_STOPPING:

        print("\n⏹️ Early stopping triggered")

        break

# =========================================
# FINAL TEST
# =========================================

print("\n📊 Loading best checkpoint...")

ckpt = torch.load(
    MODEL_SAVE_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    ckpt["model_state"]
)

test_acc, test_f1, test_auc = evaluate(
    test_loader,
    split="test"
)

print(f"\n{'='*50}")

print("🔥 FINAL TEST RESULTS")

print(
    f"Accuracy : "
    f"{test_acc:.4f} "
    f"({test_acc*100:.2f}%)"
)

print(f"F1 Score : {test_f1:.4f}")

print(f"ROC-AUC  : {test_auc:.4f}")

print(f"{'='*50}\n")