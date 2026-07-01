import os
import random
import numpy as np

import torch
import torch.nn as nn

import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from PIL import Image

# ==============================
# CONFIG
# ==============================

BASE_DIR = r"C:\Users\hanan\Desktop\datasets"

OUTPUTS_DIR = r"C:\Users\hanan\Desktop\datasets\outputs"

DATASET_PATH = rf"{BASE_DIR}\processed\video_face_sequences"

MODEL_SAVE_PATH = rf"{OUTPUTS_DIR}\models\best_video_efficientnet_lstm.pth"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 4

SEQ_LEN = 16

EPOCHS = 10

SEED = 42

# ==============================
# SEED
# ==============================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

# ==============================
# TRANSFORMS
# ==============================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# ==============================
# DATASET
# ==============================

class VideoDataset(Dataset):

    def __init__(self, root, transform):

        self.samples = []

        self.transform = transform

        for label, cls in enumerate(["real", "fake"]):

            class_path = os.path.join(root, cls)

            for vid in os.listdir(class_path):

                self.samples.append(
                    (
                        os.path.join(class_path, vid),
                        label
                    )
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        vid_path, label = self.samples[idx]

        frames = sorted([
            f for f in os.listdir(vid_path)
            if f.lower().endswith(
                ('.jpg','.jpeg','.png')
            )
        ])

        if len(frames) == 0:

            return self.__getitem__(
                (idx + 1) % len(self.samples)
            )

        frames = frames[:SEQ_LEN]

        while len(frames) < SEQ_LEN:

            frames.append(frames[-1])

        images = []

        for f in frames:

            img = Image.open(
                os.path.join(vid_path, f)
            ).convert('RGB')

            img = self.transform(img)

            images.append(img)

        return (
            torch.stack(images),
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )

# ==============================
# MODEL
# ==============================

class DeepfakeModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        self.cnn.classifier = nn.Identity()

        # freeze
        for p in self.cnn.parameters():

            p.requires_grad = False

        # unfreeze last layers
        for p in self.cnn.features[-2:].parameters():

            p.requires_grad = True

        self.lstm = nn.LSTM(
            1280,
            256,
            batch_first=True
        )

        self.fc = nn.Linear(256, 1)

    def forward(self, x):

        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)

        feat = self.cnn(x)

        feat = feat.view(B, T, -1)

        out, _ = self.lstm(feat)

        out = self.fc(out[:, -1, :])

        return out.squeeze()

# ==============================
# TRAIN / EVAL
# ==============================

def train(model, loader, optimizer, criterion, scaler):

    model.train()

    total_loss = 0

    for x, y in loader:

        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            out = model(x)

            loss = criterion(out, y)

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader):

    model.eval()

    correct, total = 0, 0

    with torch.no_grad():

        for x, y in loader:

            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)

            preds = torch.sigmoid(out) > 0.5

            correct += (preds == y).sum().item()

            total += y.size(0)

    return correct / total

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    set_seed(SEED)

    # datasets
    train_ds = VideoDataset(
        os.path.join(DATASET_PATH, "train"),
        train_transform
    )

    val_ds = VideoDataset(
        os.path.join(DATASET_PATH, "val"),
        val_transform
    )

    test_ds = VideoDataset(
        os.path.join(DATASET_PATH, "test"),
        val_transform
    )

    # loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True
    )

    # pos weight
    real = len(
        os.listdir(
            os.path.join(
                DATASET_PATH,
                "train",
                "real"
            )
        )
    )

    fake = len(
        os.listdir(
            os.path.join(
                DATASET_PATH,
                "train",
                "fake"
            )
        )
    )

    pos_weight = torch.tensor(
        [real / fake]
    ).to(DEVICE)

    # model
    model = DeepfakeModel().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2
    )

    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0

    # training loop
    for epoch in range(EPOCHS):

        loss = train(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler
        )

        val_acc = evaluate(
            model,
            val_loader
        )

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:

            best_acc = val_acc

            os.makedirs(
                os.path.dirname(MODEL_SAVE_PATH),
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH
            )

            print("✅ Best model saved")

    # test
    model.load_state_dict(
        torch.load(
            MODEL_SAVE_PATH,
            map_location=DEVICE
        )
    )

    test_acc = evaluate(
        model,
        test_loader
    )

    print(f"\n🔥 FINAL TEST ACC: {test_acc:.4f}")