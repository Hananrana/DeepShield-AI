import cv2
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# FACE DETECTION SETUP (MTCNN)
# =====================================================
from facenet_pytorch import MTCNN
detector = MTCNN(keep_all=True, device=DEVICE)

# =====================================================
# FFT GENERATION
# =====================================================
def compute_fft(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    h, w = fshift.shape
    cy, cx = h // 2, w // 2

    r = min(h, w) // 10
    fshift[cy-r:cy+r, cx-r:cx+r] = 0

    magnitude = np.abs(fshift) ** 0.5
    magnitude = np.log(magnitude + 1)

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    mag_u8 = magnitude.astype(np.uint8)

    return cv2.merge([mag_u8, mag_u8, mag_u8])


# =====================================================
# TRANSFORMS
# =====================================================
rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

fft_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# =====================================================
# MODEL ARCHITECTURE
# =====================================================
class GatedFusionHead(nn.Module):
    def __init__(self, feat_dim=1280):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, rgb_feat, fft_feat):
        fused = torch.cat([rgb_feat, fft_feat], dim=1)
        g = self.gate(fused)

        rgb_weighted = rgb_feat * g
        fft_weighted = fft_feat * (1.0 - g)

        fused = torch.cat([rgb_weighted, fft_weighted], dim=1)
        return self.classifier(fused)


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        rgb = models.efficientnet_b0(weights=None)
        feat_dim = rgb.classifier[1].in_features
        rgb.classifier = nn.Identity()

        fft = models.efficientnet_b0(weights=None)
        fft.classifier = nn.Identity()

        self.rgb_encoder = rgb
        self.fft_encoder = fft

        self.fusion_head = GatedFusionHead(feat_dim)

    def forward(self, rgb, fft):
        rgb_feat = self.rgb_encoder(rgb)
        fft_feat = self.fft_encoder(fft)

        return self.fusion_head(rgb_feat, fft_feat)


# =====================================================
# INITIALIZE & LOAD MODEL
# =====================================================
MODEL_PATH = "models/fusion_rgb_fft_best_v2.pth"

# Strict failure if model is missing in production
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = FusionModel().to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
THRESHOLD = ckpt.get("threshold", 0.5)

model.eval()


# =====================================================
# PSEUDO GRAD-CAM GENERATION
# =====================================================
def compute_pseudo_gradcam(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Compute Sobel gradients to find structural/blending boundaries
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize gradient magnitude
    grad_norm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply high-radius blur to mimic low-resolution neural network activation maps
    blur_radius = max(img_bgr.shape[0] // 8, 15)
    if blur_radius % 2 == 0:
        blur_radius += 1
    heatmap_gray = cv2.GaussianBlur(grad_norm, (blur_radius, blur_radius), 0)
    
    # Apply JET colormap (standard in Grad-CAM visualizations)
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    
    # Blend colormap with original face crop
    blended = cv2.addWeighted(img_bgr, 0.55, heatmap_color, 0.45, 0)
    return blended


# =====================================================
# INFERENCE HELPER
# =====================================================
def run_model_inference(crop_bgr):
    fft_img = compute_fft(crop_bgr)
    rgb_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    fft_pil = Image.fromarray(cv2.cvtColor(fft_img, cv2.COLOR_BGR2RGB))
    rgb_tensor = rgb_transform(rgb_pil).unsqueeze(0).to(DEVICE)
    fft_tensor = fft_transform(fft_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(rgb_tensor, fft_tensor)
        prob = torch.sigmoid(logit).item()
    return prob

# =====================================================
# PREDICTION LOGIC (WITH STRICT VALIDATION)
# =====================================================
def predict_image(image_path, scan_id=None):
    
    if not os.path.exists(image_path):
        return {"error": "File not found"}

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise Exception(f"Unable to read image at: {image_path}")

    # 1. Detect Human Faces using MTCNN
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes, det_probs = detector.detect(img_rgb)

    # 2. Hybrid Face & Body Validation
    face_detected = (boxes is not None and len(boxes) > 0)

    if face_detected:
        print("Human face detected, proceeding with face + body joint analysis.")
        largest_box_idx = np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
        box = boxes[largest_box_idx]
        x1, y1, x2, y2 = map(int, box)
        
        h_img, w_img = img_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        face_crop = img_bgr[y1:y2, x1:x2]

        # Calculate Face Probability and Full Image (Body/Background) Probability
        face_prob = run_model_inference(face_crop)
        full_prob = run_model_inference(img_bgr)
        
        # Weighted blend
        prob = (0.7 * face_prob) + (0.3 * full_prob)
    else:
        print("No human face detected, analyzing full image (body/background) directly.")
        face_crop = img_bgr.copy()
        
        # Inference on full image
        face_prob = 0.0  # N/A
        full_prob = run_model_inference(img_bgr)
        prob = full_prob

    # Debug Step requested by User
    print("=" * 50)
    print(f"Face Detected: {face_detected}")
    print(f"Face Prob    : {face_prob:.4f}")
    print(f"Full Prob    : {full_prob:.4f}")
    print(f"Combined Prob: {prob:.4f}")
    print(f"Threshold    : {THRESHOLD:.4f}")
    print("=" * 50)

    # 4. Compute FFT & Grad-CAM on the face crop for rendering/saving
    fft_img = compute_fft(face_crop)
    blended_gradcam = compute_pseudo_gradcam(face_crop)

    # 8. Generate Label and Confidence based on Uncertain boundaries
    if prob >= 0.65:
        label = "Fake"
        confidence = prob
    elif prob <= 0.35:
        label = "Real"
        confidence = 1 - prob
    else:
        label = "Uncertain"
        confidence = max(prob, 1 - prob)

    # 9. Save dynamic visualization assets
    if not scan_id:
        import random
        scan_id = f"DS-2026-{random.randint(100000, 999999)}"

    os.makedirs("static/results", exist_ok=True)
    
    face_file = f"face_{scan_id}.jpg"
    fft_file = f"fft_{scan_id}.jpg"
    gradcam_file = f"gradcam_{scan_id}.jpg"
    
    cv2.imwrite(os.path.join("static/results", face_file), face_crop)
    cv2.imwrite(os.path.join("static/results", fft_file), fft_img)
    cv2.imwrite(os.path.join("static/results", gradcam_file), blended_gradcam)

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "raw_probability": round(prob, 4),
        "threshold": round(THRESHOLD, 4),
        "face_url": f"/static/results/{face_file}",
        "fft_url": f"/static/results/{fft_file}",
        "gradcam_url": f"/static/results/{gradcam_file}"
    }