import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

# =========================
# CONFIG
# =========================

BASE_DIR = r"C:\Users\hanan\Desktop\datasets"
OUTPUTS_DIR = r"C:\Users\hanan\Desktop\outputs\models"

TRAIN_DIR = rf"{BASE_DIR}\processed\faces_haar\train"
VAL_DIR   = rf"{BASE_DIR}\processed\faces_haar\val"
TEST_DIR  = rf"{BASE_DIR}\processed\faces_haar\test"

BATCH_SIZE = 32
EPOCHS     = 15
LR         = 3e-5
PATIENCE   = 3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

BEST_PATH = rf"{OUTPUTS_DIR}\best_resnet_fast.pth"

# =========================
# TRANSFORMS (FIXED)
# =========================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =========================
# DATA
# =========================
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_ds   = datasets.ImageFolder(VAL_DIR,   transform=eval_tf)
test_ds  = datasets.ImageFolder(TEST_DIR,  transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

print("Classes:", train_ds.classes)
print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
print(f"Device: {DEVICE}")

# =========================
# MODEL (TRANSFER LEARNING)
# =========================
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze ALL layers
for param in model.parameters():
    param.requires_grad = False

# Train only last block
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace classifier
in_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 1)
)

model = model.to(DEVICE)

# =========================
# LOSS / OPTIMIZER
# =========================
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# =========================
# TRAINING
# =========================
best_val_loss = float("inf")
no_improve = 0

print("\n" + "="*55)
print(" RESNET50 FAST BASELINE TRAINING")
print("="*55)

for epoch in range(EPOCHS):

    # TRAIN
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()

        loss = criterion(model(x), y)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # VALIDATION
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.float().unsqueeze(1).to(DEVICE)

            val_loss += criterion(model(x), y).item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # SAVE BEST
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0

        torch.save(
            model.state_dict(),
            BEST_PATH
        )

        print("  ✅ Best model saved")

    else:
        no_improve += 1

        print(f"  ⚠️ No improvement ({no_improve}/{PATIENCE})")

        if no_improve >= PATIENCE:
            print("🛑 Early stopping")
            break

# =========================
# EVALUATION
# =========================
print("\n" + "="*55)
print(" RESNET50 TEST RESULTS")
print("="*55)

model.load_state_dict(
    torch.load(BEST_PATH)
)

model.eval()

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for x, y in test_loader:

        x = x.to(DEVICE)

        outputs = model(x)

        probs = torch.sigmoid(outputs).cpu().numpy()

        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs.flatten())
        all_preds.extend(preds.flatten())
        all_labels.extend(y.numpy())

# METRICS
acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec  = recall_score(all_labels, all_preds)
f1   = f1_score(all_labels, all_preds)
auc  = roc_auc_score(all_labels, all_probs)
cm   = confusion_matrix(all_labels, all_preds)

print("\n=== RESNET50 BASELINE RESULTS ===")

print(f"Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC-ROC  : {auc:.4f}")

print(f"\nConfusion Matrix:\n{cm}")

print(f"\nTrue  Real  (TN): {cm[0][0]}")
print(f"False Fake  (FP): {cm[0][1]}")
print(f"False Real  (FN): {cm[1][0]}")
print(f"True  Fake  (TP): {cm[1][1]}")

print("\n📋 MODEL SUMMARY")
print(f"ResNet50 | Acc: {acc*100:.1f}% | F1: {f1:.4f} | AUC: {auc:.4f}")