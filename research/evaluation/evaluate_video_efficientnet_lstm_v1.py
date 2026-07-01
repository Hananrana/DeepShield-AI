import os
import json
import csv
import numpy as np

from PIL import Image

import torch
import torch.nn as nn

import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================

BASE_DIR = r"C:\Users\hanan\Desktop\datasets"

OUTPUTS_DIR = r"C:\Users\hanan\Desktop\datasets\outputs"

DATASET_PATH = rf"{BASE_DIR}\processed\video_face_sequences"

MODEL_PATH = rf"{OUTPUTS_DIR}\models\best_video_efficientnet_lstm.pth"

RESULTS_DIR = rf"{OUTPUTS_DIR}\results\video_efficientnet_lstm_v1"

PLOTS_DIR = rf"{OUTPUTS_DIR}\plots\video_efficientnet_lstm_v1"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 4

SEQ_LEN = 16

print(f"\n🚀 Device: {DEVICE}")

# ==============================
# CREATE OUTPUT DIRS
# ==============================

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(PLOTS_DIR, exist_ok=True)

# ==============================
# TRANSFORMS
# ==============================

transform = transforms.Compose([
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

            raise ValueError(
                f"No frames found: {vid_path}"
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
# DATALOADER
# ==============================

test_ds = VideoDataset(
    os.path.join(DATASET_PATH, "test"),
    transform
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print(f"Test samples: {len(test_ds)}")

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
# LOAD MODEL
# ==============================

model = DeepfakeModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print(f"✅ Loaded model: {MODEL_PATH}")

# ==============================
# EVALUATION
# ==============================

all_labels = []

all_probs = []

all_preds = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(DEVICE)

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

# ==============================
# METRICS
# ==============================

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

# ==============================
# PRINT RESULTS
# ==============================

print("\n" + "=" * 50)

print("🔥 VIDEO EFFICIENTNET-LSTM V1 RESULTS")

print("=" * 50)

print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")

print(f"Precision : {prec:.4f}")

print(f"Recall    : {rec:.4f}")

print(f"F1 Score  : {f1:.4f}")

print(f"ROC-AUC   : {auc:.4f}")

print("\nConfusion Matrix:")

print(cm)

print("=" * 50)

# ==============================
# SAVE METRICS
# ==============================

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

# ==============================
# SAVE PREDICTIONS
# ==============================

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

# ==============================
# SAVE CONFUSION MATRIX
# ==============================

plt.figure(figsize=(5, 5))

plt.imshow(cm)

plt.title(
    "EfficientNet-LSTM V1 Confusion Matrix"
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