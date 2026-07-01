"""
train_fft_model.py
==================
EfficientNet-B0 binary classifier for FFT-based deepfake detection.
Supports mixed-precision training, cosine LR scheduling, class-balanced
loss, optimal threshold search, and full evaluation metrics.
"""

import os
import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Suppress GradScaler deprecation noise from newer PyTorch builds
warnings.filterwarnings(
    "ignore",
    message=".*GradScaler.*",
    category=FutureWarning,
)

# ── AMP import (compatible with PyTorch >= 1.6 and >= 2.x) ────────────────
try:
    from torch.amp import autocast, GradScaler
    _AMP_DEVICE_ARG = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _AMP_DEVICE_ARG = False

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)

# =============================================================================
# AUGMENTATION HELPER
# =============================================================================

class AddGaussianNoise:
    """Additive Gaussian noise augmentation (applied after ToTensor)."""

    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + self.std * torch.randn_like(x), 0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = r"C:\Users\hanan\Desktop\datasets"
OUTPUTS_DIR = r"C:\Users\hanan\Desktop\outputs\models"

TRAIN_DIR = rf"{BASE_DIR}\processed\fft_mtcnn\train"
VAL_DIR   = rf"{BASE_DIR}\processed\fft_mtcnn\val"
TEST_DIR  = rf"{BASE_DIR}\processed\fft_mtcnn\test"

BATCH_SIZE    = 32
EPOCHS        = 15
LR            = 2e-4
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0
LABEL_SMOOTH  = 0.05
NOISE_STD     = 0.02
IMAGE_SIZE    = 224
NUM_WORKERS   = 4

SAVE_PATH = Path(
    rf"{OUTPUTS_DIR}\fft_model_best.pth"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FFT magnitude images are NOT ImageNet-normalised
_FFT_MEAN = [0.0, 0.0, 0.0]
_FFT_STD  = [1.0, 1.0, 1.0]

torch.backends.cudnn.benchmark = True

# =============================================================================
# TRANSFORMS
# =============================================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    AddGaussianNoise(NOISE_STD),
    transforms.Normalize(_FFT_MEAN, _FFT_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_FFT_MEAN, _FFT_STD),
])

# =============================================================================
# MODEL
# =============================================================================

def build_model() -> nn.Module:
    """EfficientNet-B0 with a binary classification head."""

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    in_features: int = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 1),
    )

    return model

# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    criterion: nn.Module,
) -> float:

    model.train()

    running_loss = 0.0

    for images, labels in loader:

        images = images.to(DEVICE, non_blocking=True)

        labels = (
            labels.float()
                  .unsqueeze(1)
                  .to(DEVICE, non_blocking=True)
        )

        # Label smoothing
        labels = labels * (1.0 - 2 * LABEL_SMOOTH) + LABEL_SMOOTH

        optimizer.zero_grad(set_to_none=True)

        ctx = autocast(DEVICE) if _AMP_DEVICE_ARG else autocast()

        with ctx:
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    thresh: float = 0.5,
    find_optimal_thresh: bool = False,
):

    model.eval()

    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(DEVICE, non_blocking=True)

            logits = model(images)

            probs = torch.sigmoid(logits).cpu().view(-1)

            all_probs.append(probs)
            all_labels.append(labels)

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    best_thresh = thresh

    if find_optimal_thresh:

        best_f1 = -1.0

        for t in np.linspace(0.3, 0.7, 41):

            preds_t = (probs > t).astype(int)

            f1_t = f1_score(labels, preds_t, zero_division=0)

            if f1_t > best_f1:
                best_f1     = f1_t
                best_thresh = float(t)

    preds = (probs > best_thresh).astype(int)

    acc  = float(accuracy_score(labels, preds))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec  = float(recall_score(labels, preds, zero_division=0))
    f1   = float(f1_score(labels, preds, zero_division=0))
    auc  = float(roc_auc_score(labels, probs))
    cm   = confusion_matrix(labels, preds)

    return acc, prec, rec, f1, auc, cm, best_thresh

# =============================================================================
# REPORTING
# =============================================================================

def print_metrics(
    split: str,
    acc: float,
    prec: float,
    rec: float,
    f1: float,
    auc: float,
    cm,
    thresh: float,
) -> None:

    tn, fp, fn, tp = cm.ravel()

    sep = "=" * 55

    log.info(sep)
    log.info(f"  {split}  (threshold = {thresh:.3f})")
    log.info(sep)

    log.info(f"  Accuracy  : {acc:.4f}  ({acc * 100:.1f}%)")
    log.info(f"  Precision : {prec:.4f}")
    log.info(f"  Recall    : {rec:.4f}")
    log.info(f"  F1 Score  : {f1:.4f}")
    log.info(f"  AUC-ROC   : {auc:.4f}")

    log.info("")
    log.info(f"  Confusion Matrix:\n{cm}")
    log.info("")

    log.info(f"  True  Real  (TN) : {tn}")
    log.info(f"  False Fake  (FP) : {fp}")
    log.info(f"  False Real  (FN) : {fn}")
    log.info(f"  True  Fake  (TP) : {tp}")

    log.info(sep)

    log.info(
        f"  MODEL SUMMARY | Acc: {acc * 100:.1f}% | "
        f"Precision: {prec:.4f} | Recall: {rec:.4f} | "
        f"F1: {f1:.4f} | AUC: {auc:.4f}"
    )

    log.info(sep)

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    # Validate directories
    for split, path in [
        ("Train", TRAIN_DIR),
        ("Val", VAL_DIR),
        ("Test", TEST_DIR)
    ]:

        if not Path(path).is_dir():
            log.error(f"{split} directory not found: {path}")
            sys.exit(1)

    # Datasets & loaders
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=eval_transform)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=eval_transform)

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    log.info(f"Classes : {train_ds.classes}")
    log.info(f"Train   : {len(train_ds):,} samples")
    log.info(f"Val     : {len(val_ds):,} samples")
    log.info(f"Test    : {len(test_ds):,} samples")
    log.info(f"Device  : {DEVICE}")

    # Class balance
    real_idx   = train_ds.class_to_idx.get("real", 0)

    real_count = sum(
        1 for _, lbl in train_ds.samples if lbl == real_idx
    )

    fake_count = len(train_ds) - real_count

    pos_weight_val = fake_count / max(real_count, 1)

    pos_weight = torch.tensor(
        [pos_weight_val],
        device=DEVICE
    )

    log.info(
        f"Class balance -> Real: {real_count:,} | "
        f"Fake: {fake_count:,} | "
        f"pos_weight: {pos_weight_val:.3f}"
    )

    # Model / optimiser / scheduler
    model = build_model().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    scaler = GradScaler()

    # Training loop
    best_f1     = 0.0
    best_thresh = 0.5

    log.info("Starting training ...\n")

    for epoch in range(1, EPOCHS + 1):

        find_opt = epoch >= EPOCHS - 2

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion
        )

        acc, prec, rec, f1, auc, cm, opt_thresh = evaluate(
            model,
            val_loader,
            thresh=best_thresh,
            find_optimal_thresh=find_opt,
        )

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        log.info(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"Prec: {prec:.4f} | "
            f"Rec: {rec:.4f} | "
            f"F1: {f1:.4f} | "
            f"AUC: {auc:.4f} | "
            f"LR: {lr:.2e}"
        )

        if f1 > best_f1:

            best_f1     = f1
            best_thresh = opt_thresh

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "threshold": opt_thresh,
                    "epoch": epoch,
                    "val_f1": f1,
                    "val_auc": auc,
                },
                SAVE_PATH,
            )

            log.info(
                f"  [SAVED] Best model  "
                f"(F1={f1:.4f}, thresh={opt_thresh:.3f})"
            )

    # Final test evaluation
    log.info("\nLoading best checkpoint for test evaluation ...")

    ckpt = torch.load(
        SAVE_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(ckpt["model_state"])

    test_thresh = float(ckpt["threshold"])

    acc, prec, rec, f1, auc, cm, _ = evaluate(
        model,
        test_loader,
        thresh=test_thresh
    )

    print_metrics(
        split="EFFICIENTNET-B0 TEST RESULTS",
        acc=acc,
        prec=prec,
        rec=rec,
        f1=f1,
        auc=auc,
        cm=cm,
        thresh=test_thresh,
    )

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    import torch.multiprocessing as mp

    mp.freeze_support()

    main()