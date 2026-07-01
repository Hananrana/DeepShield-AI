import cv2
from facenet_pytorch import MTCNN
import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
detector = MTCNN(keep_all=True, device=DEVICE)

img_bgr = cv2.imread("test.jpg")
if img_bgr is None:
    print("Could not read test.jpg")
else:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes, probs = detector.detect(img_rgb)
    print("Boxes:", boxes)
    print("Probs:", probs)
    if boxes is not None and len(boxes) > 0:
        largest_box_idx = np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
        box = boxes[largest_box_idx]
        x1, y1, x2, y2 = map(int, box)
        print(f"Largest Face Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
