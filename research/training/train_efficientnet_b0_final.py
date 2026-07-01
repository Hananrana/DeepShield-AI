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
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 4
LR_STAGE1 = 1e-4
LR_STAGE2 = 1e-5
PATIENCE = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BEST_PATH = rf"{OUTPUTS_DIR}\best_efficientnet_final.pth"

# =========================
# AUGMENTATION
# =========================
train_tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224, scale=(0.85,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3,0.3,0.3),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

eval_tf = transforms.Compose([
    transforms.Resize((224,224)),
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

# 🔥 FIX: num_workers=0 (Windows safe)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=0)

print("Classes:", train_ds.classes)
print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
print(f"Device: {DEVICE}")

# =========================
# MODEL
# =========================
model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)

# Partial freeze
for param in model.features[:4].parameters():
    param.requires_grad = False
for param in model.features[4:].parameters():
    param.requires_grad = True

# Classifier
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 1)
)

model = model.to(DEVICE)

# =========================
# LOSS / OPTIM
# =========================
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_STAGE1
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.3, patience=2
)

# =========================
# TRAIN FUNCTIONS
# =========================
def train_one_epoch():
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate():
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.float().unsqueeze(1).to(DEVICE)

            total_loss += criterion(model(x), y).item()

    return total_loss / len(val_loader)

# =========================
# STAGE 1
# =========================
best_val = float("inf")
no_improve = 0

print("\n=== Stage 1 Training ===")

for epoch in range(EPOCHS_STAGE1):

    train_loss = train_one_epoch()
    val_loss   = validate()

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        no_improve = 0

        torch.save(
            model.state_dict(),
            BEST_PATH
        )

        print("  ✅ Best saved")

    else:
        no_improve += 1

        print(f"  ⚠️ No improve ({no_improve}/{PATIENCE})")

        if no_improve >= PATIENCE:
            print("🛑 Early stop Stage 1")
            break

# =========================
# STAGE 2 (CONTROLLED)
# =========================
print("\n=== Stage 2 Fine-Tuning ===")

for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR_STAGE2
)

for epoch in range(EPOCHS_STAGE2):

    train_loss = train_one_epoch()
    val_loss   = validate()

    print(f"Fine Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss

        torch.save(
            model.state_dict(),
            BEST_PATH
        )

        print("  ✅ Improved & saved")

    else:
        print("  ⚠️ No improvement")

# =========================
# TEST
# =========================
print("\n=== FINAL TEST RESULTS ===")

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

        preds = (probs > 0.6).astype(int)

        all_probs.extend(probs.flatten())
        all_preds.extend(preds.flatten())
        all_labels.extend(y.numpy())

acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec  = recall_score(all_labels, all_preds)
f1   = f1_score(all_labels, all_preds)
auc  = roc_auc_score(all_labels, all_probs)
cm   = confusion_matrix(all_labels, all_preds)

print(f"\nAccuracy : {acc:.4f} ({acc*100:.1f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC-ROC  : {auc:.4f}")

print(f"\nConfusion Matrix:\n{cm}")

print(f"\nTN: {cm[0][0]} | FP: {cm[0][1]}")
print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")

print("\n📋 EfficientNet FINAL MODEL (STABLE + FIXED)")