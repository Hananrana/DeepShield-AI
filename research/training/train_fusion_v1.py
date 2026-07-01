
import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

warnings.filterwarnings("ignore", message=".*GradScaler.*", category=FutureWarning)

# ── AMP: support both PyTorch 1.x and 2.x ─────────────────────────────────
try:
    from torch.amp import autocast, GradScaler          # PyTorch >= 2.0
    _AMP_DEVICE = "cuda"
    _AMP_DEVICE_ARG = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler     # PyTorch 1.x
    _AMP_DEVICE = None
    _AMP_DEVICE_ARG = False

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = r"C:\Users\hanan\Desktop\datasets"
OUTPUTS_DIR = r"C:\Users\hanan\Desktop\outputs\models"

RGB_ROOT = rf"{BASE_DIR}\processed\faces_haar"
FFT_ROOT = rf"{BASE_DIR}\processed\fft_mtcnn"

BATCH_SIZE   = 32
EPOCHS       = 15
LR           = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
LABEL_SMOOTH = 0.05
IMAGE_SIZE   = 224
NUM_WORKERS  = 4

SAVE_PATH = Path(
    rf"{OUTPUTS_DIR}\fusion_rgb_fft_best.pth"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark = True

# =============================================================================
# TRANSFORMS
# =============================================================================

rgb_train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

rgb_eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# FFT magnitude images are NOT ImageNet-normalised
fft_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
])


# =============================================================================
# DATASET  (paired RGB + FFT)
# =============================================================================

class DualDataset(Dataset):
    """
    Loads matched RGB and FFT image pairs from parallel directory trees.
    Only samples where BOTH rgb and fft files exist are included.
    """

    def __init__(
        self,
        rgb_root: str,
        fft_root: str,
        split: str,
        transform_rgb,
        transform_fft,
    ):
        rgb_split = Path(rgb_root) / split
        fft_split = Path(fft_root) / split

        self.samples: list[tuple[Path, Path, int]] = []
        missing = 0

        for label_str, label_int in [("real", 0), ("fake", 1)]:
            rgb_label_dir = rgb_split / label_str
            fft_label_dir = fft_split / label_str

            if not rgb_label_dir.is_dir():
                log.warning(f"Missing RGB dir: {rgb_label_dir}")
                continue

            for rgb_path in sorted(rgb_label_dir.iterdir()):
                if not rgb_path.is_file():
                    continue
                fft_path = fft_label_dir / rgb_path.name
                if fft_path.exists():
                    self.samples.append((rgb_path, fft_path, label_int))
                else:
                    missing += 1

        if missing:
            log.warning(
                f"[{split}] Skipped {missing} RGB files with no matching FFT pair."
            )

        self.transform_rgb = transform_rgb
        self.transform_fft = transform_fft

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rgb_path, fft_path, label = self.samples[idx]

        rgb_img = Image.open(rgb_path).convert("RGB")
        fft_img = Image.open(fft_path).convert("RGB")

        rgb_tensor = self.transform_rgb(rgb_img)
        fft_tensor = self.transform_fft(fft_img)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return rgb_tensor, fft_tensor, label_tensor


# =============================================================================
# MODEL  (Dual-branch EfficientNet-B0 with MLP fusion head)
# =============================================================================

class FusionModel(nn.Module):
    """
    Two independent EfficientNet-B0 encoders whose feature vectors are
    concatenated and passed through a small MLP classifier.
    """

    def __init__(self):
        super().__init__()

        # RGB branch ──────────────────────────────────────────────
        rgb_backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        rgb_feat_dim: int = rgb_backbone.classifier[1].in_features
        rgb_backbone.classifier = nn.Identity()
        self.rgb_encoder = rgb_backbone

        # FFT branch ──────────────────────────────────────────────
        fft_backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        fft_feat_dim: int = fft_backbone.classifier[1].in_features
        fft_backbone.classifier = nn.Identity()
        self.fft_encoder = fft_backbone

        fused_dim = rgb_feat_dim + fft_feat_dim    # 1280 + 1280 = 2560

        # Fusion MLP head ─────────────────────────────────────────
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1),
        )

    def forward(self, rgb: torch.Tensor, fft: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb_encoder(rgb)    # (B, 1280)
        fft_feat = self.fft_encoder(fft)    # (B, 1280)
        fused    = torch.cat([rgb_feat, fft_feat], dim=1)   # (B, 2560)
        return self.fusion_head(fused)      # (B, 1)


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

    for rgb, fft, labels in loader:
        rgb    = rgb.to(DEVICE, non_blocking=True)
        fft    = fft.to(DEVICE, non_blocking=True)
        labels = (
            labels.unsqueeze(1).to(DEVICE, non_blocking=True)
        )
        # Label smoothing
        labels = labels * (1.0 - 2 * LABEL_SMOOTH) + LABEL_SMOOTH

        optimizer.zero_grad(set_to_none=True)

        ctx = autocast(_AMP_DEVICE) if _AMP_DEVICE_ARG else autocast()
        with ctx:
            logits = model(rgb, fft)
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
    """
    Returns
    -------
    (accuracy, precision, recall, f1, auc, confusion_matrix, best_threshold)
    """
    model.eval()
    all_probs:  list = []
    all_labels: list = []

    with torch.no_grad():
        for rgb, fft, labels in loader:
            rgb = rgb.to(DEVICE, non_blocking=True)
            fft = fft.to(DEVICE, non_blocking=True)

            logits = model(rgb, fft)
            probs  = torch.sigmoid(logits).cpu().view(-1)
            all_probs.append(probs)
            all_labels.append(labels)

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    best_thresh = thresh

    if find_optimal_thresh:
        best_f1 = -1.0
        for t in np.linspace(0.3, 0.7, 41):
            preds_t = (probs > t).astype(int)
            f1_t    = f1_score(labels, preds_t, zero_division=0)
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
    sep = "=" * 58
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
        f"  FUSION MODEL SUMMARY | Acc: {acc * 100:.1f}% | "
        f"Precision: {prec:.4f} | Recall: {rec:.4f} | "
        f"F1: {f1:.4f} | AUC: {auc:.4f}"
    )
    log.info(sep)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    # ── Validate root directories ──────────────────────────────────────────
    for name, root in [("RGB", RGB_ROOT), ("FFT", FFT_ROOT)]:
        if not Path(root).is_dir():
            log.error(f"{name} root directory not found: {root}")
            sys.exit(1)

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = DualDataset(RGB_ROOT, FFT_ROOT, "train", rgb_train_transform, fft_transform)
    val_ds   = DualDataset(RGB_ROOT, FFT_ROOT, "val",   rgb_eval_transform,  fft_transform)
    test_ds  = DualDataset(RGB_ROOT, FFT_ROOT, "test",  rgb_eval_transform,  fft_transform)

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    log.info(f"Train : {len(train_ds):,} paired samples")
    log.info(f"Val   : {len(val_ds):,} paired samples")
    log.info(f"Test  : {len(test_ds):,} paired samples")
    log.info(f"Device: {DEVICE}")

    if len(train_ds) == 0:
        log.error(
            "Training set is empty. Check that RGB and FFT filenames match "
            "and that the split directories (train/val/test) exist."
        )
        sys.exit(1)

    # ── Class balance -> pos_weight for BCEWithLogitsLoss ──────────────────
    labels_list = [lbl for *_, lbl in train_ds.samples]
    fake_count  = sum(labels_list)
    real_count  = len(labels_list) - fake_count
    pos_weight_val = fake_count / max(real_count, 1)
    pos_weight = torch.tensor([pos_weight_val], device=DEVICE)

    log.info(
        f"Class balance -> Real: {real_count:,} | Fake: {fake_count:,} | "
        f"pos_weight: {pos_weight_val:.3f}"
    )

    # ── Model / optimiser / scheduler ─────────────────────────────────────
    model     = FusionModel().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {total_params:,}")

    # ── Training loop ─────────────────────────────────────────────────────
    best_f1     = 0.0
    best_thresh = 0.5

    log.info("Starting training ...\n")

    for epoch in range(1, EPOCHS + 1):
        find_opt   = epoch >= EPOCHS - 2       # search optimal threshold in last 3 epochs
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)

        acc, prec, rec, f1, auc, cm, opt_thresh = evaluate(
            model, val_loader,
            thresh=best_thresh,
            find_optimal_thresh=find_opt,
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        log.info(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | "
            f"F1: {f1:.4f} | AUC: {auc:.4f} | LR: {lr:.2e}"
        )

        if f1 > best_f1:
            best_f1     = f1
            best_thresh = opt_thresh
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "threshold":   opt_thresh,
                    "epoch":       epoch,
                    "val_f1":      f1,
                    "val_auc":     auc,
                },
                SAVE_PATH,
            )
            log.info(f"  [SAVED] Best model  (F1={f1:.4f}, thresh={opt_thresh:.3f})")

    # ── Final test evaluation ──────────────────────────────────────────────
    log.info("\nLoading best checkpoint for test evaluation ...")
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    test_thresh = float(ckpt["threshold"])

    acc, prec, rec, f1, auc, cm, _ = evaluate(
        model, test_loader, thresh=test_thresh
    )

    print_metrics(
        split="DUAL RGB+FFT FUSION — TEST RESULTS",
        acc=acc, prec=prec, rec=rec, f1=f1,
        auc=auc, cm=cm, thresh=test_thresh,
    )


# =============================================================================
# ENTRY POINT  (Windows multiprocessing fix)
# =============================================================================

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()