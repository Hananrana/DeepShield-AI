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

from sklearn.metrics import f1_score

# ==============================
# CONFIG
# ==============================

BASE_DIR = r"C:\Users\hanan\Desktop\datasets"

OUTPUTS_DIR = r"C:\Users\hanan\Desktop\datasets\outputs"

DATASET_PATH = rf"{BASE_DIR}\processed\video_face_sequences"

MODEL_SAVE_PATH = rf"{OUTPUTS_DIR}\models\best_video_efficientnet_lstm_v2.pth"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 4

SEQ_LEN = 16

EPOCHS = 15

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

        # RANDOM SAMPLING
        if len(frames) >= SEQ_LEN:

            frames = sorted(
                random.sample(frames, SEQ_LEN)
            )

        else:

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

        for p in self.cnn.parameters():

            p.requires_grad = False

        # UNFREEZE MORE LAYERS
        for p in self.cnn.features[-4:].parameters():

            p.requires_grad = True

        self.lstm = nn.LSTM(
            1280,
            256,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(256, 1)

    def forward(self, x):

        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)

        feat = self.cnn(x)

        feat = feat.view(B, T, -1)

        out, _ = self.lstm(feat)

        out = self.dropout(out[:, -1, :])

        out = self.fc(out)

        return out.squeeze()

# ==============================
# TRAIN
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

# ==============================
# EVALUATE
# ==============================

def evaluate_full(model, loader):

    model.eval()

    all_preds = []

    all_labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)

            out = model(x)

            probs = torch.sigmoid(
                out
            ).cpu().numpy()

            all_preds.extend(probs)

            all_labels.extend(y.numpy())

    best_f1, best_t = 0, 0.5

    for t in np.linspace(0.3, 0.7, 21):

        preds = (
            np.array(all_preds) > t
        ).astype(int)

        f1 = f1_score(
            all_labels,
            preds
        )

        if f1 > best_f1:

            best_f1 = f1

            best_t = t

    acc = (
        (np.array(all_preds) > best_t)
        == np.array(all_labels)
    ).mean()

    return acc, best_f1, best_t

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    set_seed(SEED)

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

    model = DeepfakeModel().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # DIFFERENTIAL LR
    cnn_params = []

    head_params = []

    for name, p in model.named_parameters():

        if p.requires_grad:

            if "cnn" in name:

                cnn_params.append(p)

            else:

                head_params.append(p)

    optimizer = torch.optim.Adam([
        {"params": cnn_params, "lr": 1e-5},
        {"params": head_params, "lr": 5e-5}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    scaler = torch.cuda.amp.GradScaler()

    best_f1 = 0

    for epoch in range(EPOCHS):

        loss = train(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler
        )

        val_acc, val_f1, best_t = evaluate_full(
            model,
            val_loader
        )

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"F1: {val_f1:.4f} | "
            f"T: {best_t:.2f}"
        )

        if val_f1 > best_f1:

            best_f1 = val_f1

            os.makedirs(
                os.path.dirname(MODEL_SAVE_PATH),
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH
            )

            print("✅ Best model saved")

    model.load_state_dict(
        torch.load(
            MODEL_SAVE_PATH,
            map_location=DEVICE
        )
    )

    test_acc, test_f1, _ = evaluate_full(
        model,
        test_loader
    )

    print(
        f"\n🔥 FINAL TEST ACC: "
        f"{test_acc:.4f} | "
        f"F1: {test_f1:.4f}"
    )