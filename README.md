<div align="center">

# 🛡️ DeepShield AI

### AI-Powered Forensic Platform for Detecting AI-Generated Synthetic Images & Deepfake Videos

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)]
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

**Detecting AI-generated synthetic images and deepfake videos using RGB analysis, Fourier Frequency Spectrum (FFT), and Deep Learning.**

</div>

---

# 📖 Overview

DeepShield AI is an AI-powered digital forensic platform developed as a Final Year Project (FYP) to detect AI-generated synthetic images and deepfake videos.

The system combines **spatial image analysis (RGB)** with **frequency-domain analysis (FFT)** to identify manipulation artifacts that are often invisible to the human eye.

Unlike traditional image classifiers, DeepShield provides explainable forensic evidence through:

- FFT Spectrum Analysis
- Face Detection
- Grad-CAM Visualization
- Confidence Scores
- Security Risk Assessment
- Forensic Reports

---

# ✨ Features

## 🖼 Image Deepfake Detection

- Human face validation
- RGB feature extraction
- FFT frequency analysis
- RGB + FFT Fusion Model
- Confidence prediction
- Forensic evidence visualization

---

## 🎥 Video Deepfake Detection

- Frame extraction
- Face detection
- Temporal sequence analysis
- EfficientNet + LSTM pipeline
- Deepfake probability estimation

---

## 🔬 Forensic Analysis

- FFT Spectrum
- Face Crop
- Grad-CAM Heatmap
- Confidence Score
- Risk Verdict
- JSON Export
- PDF Report

---

## 🌐 Web Application

- Flask Backend
- Responsive UI
- Authentication
- Admin Dashboard
- Image Upload
- Video Upload
- Report Generation

---

# 🏗 System Architecture

```text
                     Input Image / Video
                              │
                              ▼
                    Face Detection (MTCNN)
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
         RGB Processing              FFT Processing
                │                           │
          EfficientNet-B0            EfficientNet-B0
                │                           │
                └─────────────┬─────────────┘
                              ▼
                    Gated RGB-FFT Fusion
                              │
                              ▼
                     Binary Classification
                              │
                              ▼
          GradCAM • FFT • Confidence • Verdict
```

---

# 🧠 Deep Learning Models

| Model | Purpose |
|---------|----------|
| CNN Baseline | Performance Baseline |
| ResNet50 | Spatial Feature Extraction |
| EfficientNet-B0 | Image Classification |
| FFT EfficientNet | Frequency Analysis |
| RGB + FFT Fusion V1 | Feature Fusion |
| RGB + FFT Fusion V2 | Gated Feature Fusion |
| EfficientNet + LSTM | Video Deepfake Detection |
| Xception + BiLSTM | Temporal Video Analysis |

---

# 🛠 Technology Stack

### Programming

- Python
- JavaScript
- HTML5
- CSS3

### AI & Deep Learning

- PyTorch
- TorchVision
- OpenCV
- NumPy
- Pillow
- MTCNN

### Backend

- Flask
- SQLite

### Frontend

- HTML
- CSS
- JavaScript

---

# 📂 Repository Structure

```text
DeepShield-AI
│
├── app/
│   ├── experiments/
│   ├── inference/
│   ├── static/
│   ├── templates/
│   ├── tests/
│   ├── app.py
│   ├── db.py
│   ├── migrate.py
│   └── requirements.txt
│
├── docs/
│   ├── experiments/
│   └── impact_analysis.md
│
├── research/
│   ├── training/
│   └── evaluation/
│
├── README.md
├── LICENSE
└── .gitignore
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Hananrana/DeepShield-AI.git
```

Go into the project

```bash
cd DeepShield-AI
```

Install dependencies

```bash
pip install -r app/requirements.txt
```

Run the application

```bash
cd app

python app.py
```

Open

```
http://127.0.0.1:5000
```

---

# 📊 Research Experiments

This repository contains multiple experiments performed during model development.

### Image Models

- CNN Baseline
- ResNet50
- EfficientNet-B0
- FFT EfficientNet
- RGB + FFT Fusion V1
- RGB + FFT Fusion V2

### Video Models

- RGB Video Detector
- EfficientNet + LSTM
- EfficientNet + LSTM V2
- EfficientNet + LSTM Final
- Xception + BiLSTM + Attention

Evaluation scripts and training scripts are available in the `research` directory.

---

# 📈 Results

The project evaluates multiple models using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

Experiment outputs, logs, plots, and evaluation metrics are included in the `docs/experiments` directory.

---

# 🔮 Future Improvements

- Vision Transformer (ViT)
- Swin Transformer
- CLIP-based Detection
- Diffusion Model Detection
- Explainable AI (True Grad-CAM)
- ONNX/TensorRT Optimization
- Cloud Deployment
- REST API
- Mobile Application

---

# 👨‍💻 Author

## Hanan Ali Khan

Software Engineering Student

Specializations:

- Artificial Intelligence
- Computer Vision
- Deep Learning
- Full-Stack Development

GitHub:
https://github.com/Hananrana

---

# 📄 License

This project is licensed under the MIT License.

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a Star!

Made with ❤️ using Python, PyTorch and Flask.

</div>
