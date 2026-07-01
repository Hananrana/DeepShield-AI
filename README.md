<div align="center">

# 🛡️ DeepShield AI

### AI-Powered Forensic Platform for Detecting AI-Generated Synthetic Images & Deepfake Videos

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)]
[![SQLite](https://img.shields.io/badge/SQLite-Database-blue?logo=sqlite)]
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</p>

---

### Detecting AI-generated synthetic images and deepfake videos using RGB analysis, Fourier Frequency Spectrum (FFT), Explainable AI, and Deep Learning.

**Final Year Project (FYP)**  
**Department of Software Engineering**

</div>

---

# 📖 Overview

DeepShield AI is an enterprise-inspired AI-powered digital forensic platform developed as a Final Year Project (FYP) for detecting AI-generated synthetic images and deepfake videos.

The platform combines modern computer vision techniques with deep learning to identify manipulation artifacts that are often invisible to the human eye.

Unlike conventional binary classifiers, DeepShield provides an explainable forensic workflow by combining:

- RGB Spatial Analysis
- Fourier Frequency Spectrum (FFT)
- Face Localization
- Deep Learning Classification
- Explainable AI Visualization
- Confidence Scoring
- Risk Assessment
- Forensic Report Generation

The project includes both **image forensic analysis** and **deepfake video detection** within a unified Flask web application.

---

# ✨ Key Features

## 🖼 Image Deepfake Detection

- Human face validation
- MTCNN face localization
- RGB feature extraction
- FFT spectrum analysis
- RGB + FFT Fusion
- Confidence estimation
- Explainable AI visualization
- Authenticity prediction

---

## 🎥 Video Deepfake Detection

- Video upload
- Frame extraction
- Face tracking
- Temporal sequence analysis
- EfficientNet + LSTM inference
- Deepfake probability estimation

---

## 🔬 Forensic Analysis

DeepShield automatically generates forensic evidence including:

- Face Crop
- FFT Spectrum
- Grad-CAM Visualization
- Confidence Score
- Threat Level
- Risk Assessment
- Scan ID
- Timestamp
- PDF Report
- JSON Report

---

## 🌐 Enterprise Web Platform

The web application provides:

- Secure Authentication
- User Registration
- Password Recovery
- Admin Login
- Responsive Dashboard
- Image Detection Module
- Video Detection Module
- PDF Report Generation
- JSON Export
- Security Audit Logs
- Login History
- User Management

---

# 🚀 Why DeepShield?

Traditional AI image detectors generally provide only a binary prediction.

DeepShield extends this concept by integrating explainable forensic analysis and enterprise-level reporting.

| Traditional Detector | DeepShield AI |
|----------------------|---------------|
| Binary Prediction | ✅ Explainable Prediction |
| RGB Only | ✅ RGB + FFT Fusion |
| No Frequency Analysis | ✅ FFT Spectrum |
| Limited Visualization | ✅ Grad-CAM |
| No Reports | ✅ PDF Reports |
| No Admin Dashboard | ✅ Enterprise Dashboard |
| No User Authentication | ✅ Secure Authentication |
| Image Only | ✅ Image + Video Detection |

---

# 🧠 Deep Learning Models

During research, multiple architectures were developed and evaluated.

## Image Detection

| Model | Purpose |
|---------|----------|
| CNN Baseline | Baseline Performance |
| ResNet50 | RGB Feature Extraction |
| EfficientNet-B0 | Image Classification |
| FFT EfficientNet | Frequency Domain Analysis |
| RGB + FFT Fusion V1 | Dual Feature Fusion |
| RGB + FFT Fusion V2 | Gated Feature Fusion (Final Model) |

---

## Video Detection

| Model | Purpose |
|---------|----------|
| RGB Video Detector | Baseline |
| EfficientNet + LSTM | Temporal Detection |
| EfficientNet + LSTM V2 | Improved Sequence Learning |
| EfficientNet + LSTM Final | Final Video Model |
| Xception + BiLSTM + Attention | Comparative Research |

---

# 🛠 Technology Stack

## Programming Languages

- Python
- JavaScript
- HTML5
- CSS3

---

## Artificial Intelligence

- PyTorch
- TorchVision
- OpenCV
- NumPy
- Pillow
- MTCNN
- EfficientNet
- LSTM
- CNN

---

## Backend

- Flask
- SQLite

---

## Frontend

- HTML5
- CSS3
- JavaScript

---

## Development Tools

- VS Code
- Git
- GitHub
- Jupyter Notebook
- Google Colab

---


# 📸 Application Screenshots

## 🔐 User Authentication

### Login Interface

![Login Page](docs/screenshots/01_login_page.png)

Secure user authentication with a modern enterprise-inspired interface.

---

### Admin Login

![Admin Login](docs/screenshots/02_admin_login.png)

Dedicated administrator authentication portal with role-based access.

---

### User Registration

![Signup](docs/screenshots/03_signup_page.png)

Create a secure DeepShield account with validation and password policies.

---

# 🏠 Homepage

### Landing Page

![Homepage](docs/screenshots/04_homepage.png)

Modern landing page introducing DeepShield AI and its forensic capabilities.

---

### Supported AI Generator Detection

![Generator Support](docs/screenshots/05_generator_support.png)

DeepShield is designed to analyze media generated or manipulated using modern AI systems through forensic artifact detection techniques.

---

### Real vs Synthetic Comparison

![Comparison](docs/screenshots/06_real_vs_synthetic_comparison.png)

Visual comparison demonstrating authentic and AI-generated media analysis.

---

# 🖼 Image Forensics Module

### Image Detection Workspace

![Image Lab](docs/screenshots/07_image_forensics_lab.png)

Upload images and perform RGB-FFT forensic analysis.

---

# 🎥 Video Forensics Module

### Video Detection Workspace

![Video Lab](docs/screenshots/08_video_forensics_lab.png)

Analyze deepfake videos using temporal deep learning models.

---

# 🏗 System Architecture

### Detection Pipeline

![Pipeline](docs/screenshots/09_detection_pipeline.png)

DeepShield follows a multi-stage AI forensic pipeline from media ingestion to explainable prediction.

---

# 📊 Platform Overview

### Why DeepShield?

![Platform Comparison](docs/screenshots/10_platform_comparison.png)

Comparison between conventional AI detectors and the DeepShield forensic platform.

---

### Operational Status Dashboard

![Operational Status](docs/screenshots/11_operational_status.png)

Real-time monitoring of inference engine, preprocessing pipeline, and system health.

---

### Authenticated Homepage

![Authenticated Homepage](docs/screenshots/12_authenticated_homepage.png)

Homepage after successful user authentication with secure navigation.

---

# 👨‍💼 Administration

### User Management

![Admin Accounts](docs/screenshots/13_admin_accounts.png)

Manage registered users and monitor account status.

---

### Login History

![Login History](docs/screenshots/14_admin_login_history.png)

Monitor authentication activities, timestamps, and IP addresses.

---

### Security Logs

![Security Logs](docs/screenshots/15_admin_security_logs.png)

Comprehensive audit trail for platform security events.

---

# 🧪 Detection Results

## AI Generated Image Detection

### Synthetic Image

![Fake Image](docs/screenshots/16_image_detection_result_fake.png)

Complete forensic report showing AI-generated image detection with confidence score, FFT spectrum, Grad-CAM visualization, and downloadable reports.

---

### Authentic Image

![Real Image](docs/screenshots/17_image_detection_result_real.png)

Example of successful authentic image verification.

---

## Video Deepfake Detection

### Authentic Video

![Real Video](docs/screenshots/18_video_detection_result_real.png)

Video successfully verified as authentic.

---

### Synthetic Video

![Fake Video](docs/screenshots/19_video_detection_result_fake.png)

Deepfake video detection with forensic evidence.

---

# 📄 Forensic Reports

### Real Video PDF Report

![Real PDF](docs/screenshots/20_pdf_forensic_report_real_video.png)

Automatically generated forensic report suitable for documentation and investigation.

---

### Synthetic Video PDF Report

![Fake PDF](docs/screenshots/21_pdf_forensic_report_fake_video.png)

Professional PDF report containing detection verdict, confidence score, and forensic metadata.

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
      Grad-CAM • FFT • Confidence • Verdict
```

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
│   │   ├── logs/
│   │   ├── plots/
│   │   └── results/
│   │
│   ├── screenshots/
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

## Clone Repository

```bash
git clone https://github.com/Hananrana/DeepShield-AI.git
```

---

## Enter Project

```bash
cd DeepShield-AI
```

---

## Install Dependencies

```bash
pip install -r app/requirements.txt
```

---

## Run Application

```bash
cd app

python app.py
```

---

Open your browser:

```
http://127.0.0.1:5000
```

---
# 📊 Research Experiments

DeepShield AI was developed through multiple research experiments to identify the most effective architecture for AI-generated media detection.

## 🖼 Image Detection Models

- CNN Baseline
- ResNet50
- EfficientNet-B0
- FFT EfficientNet
- RGB + FFT Fusion V1
- RGB + FFT Fusion V2 (Final)

---

## 🎥 Video Detection Models

- RGB Video Detector
- EfficientNet + LSTM
- EfficientNet + LSTM V2
- EfficientNet + LSTM Final
- Xception + BiLSTM + Attention

---

Training scripts can be found in:

```text
research/training/
```

Evaluation scripts are available in:

```text
research/evaluation/
```

---

# 📈 Experimental Results

Multiple deep learning architectures were evaluated throughout the development process.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Prediction Confidence
- FFT Spectrum Analysis

The repository contains:

- Training Scripts
- Evaluation Scripts
- Confusion Matrices
- Prediction Results
- Performance Metrics
- Experiment Logs

located in:

```text
docs/experiments/
```

---

# 🔬 Detection Pipeline

DeepShield follows a multi-stage forensic workflow:

```text
Image / Video
      │
      ▼
Face Detection
      │
      ▼
RGB Analysis
      │
      ├──────────────┐
      │              │
      ▼              ▼
FFT Generation   Spatial Features
      │              │
      └──────┬───────┘
             ▼
      Feature Fusion
             ▼
 EfficientNet Classifier
             ▼
 Confidence Estimation
             ▼
 Grad-CAM Visualization
             ▼
 PDF & JSON Report
```

---

# 📁 Repository Highlights

✔ Research Code

✔ Training Scripts

✔ Evaluation Scripts

✔ Flask Web Application

✔ Responsive User Interface

✔ Image Deepfake Detection

✔ Video Deepfake Detection

✔ Enterprise Dashboard

✔ Authentication System

✔ PDF Report Generation

✔ JSON Export

✔ Security Logs

✔ Login History

✔ AI Forensic Analysis

---

# 🎯 Project Objectives

- Detect AI-generated synthetic images
- Detect manipulated videos
- Reduce misinformation risks
- Improve digital media verification
- Provide explainable AI predictions
- Generate forensic evidence automatically
- Deliver an enterprise-ready detection platform

---

# 🔮 Future Improvements

The project can be extended with:

- Vision Transformer (ViT)
- Swin Transformer
- ConvNeXt
- CLIP-based Detection
- Diffusion Model Detection
- Explainable AI (True Grad-CAM)
- ONNX Runtime Optimization
- TensorRT Deployment
- REST API
- Docker Deployment
- Kubernetes Support
- Mobile Application
- Cloud Deployment
- Multi-language Interface
- Batch Processing
- Live Camera Detection

---

# 🤝 Contributing

Contributions are welcome.

If you would like to improve DeepShield AI:

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/NewFeature
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push

```bash
git push origin feature/NewFeature
```

5. Open a Pull Request

---

# 📚 References

This project was developed using concepts and methodologies from:

- PyTorch Documentation
- OpenCV Documentation
- TorchVision
- EfficientNet
- FaceForensics++
- DeepFake Detection Challenge (DFDC)
- MTCNN Face Detection
- Fourier Transform Analysis

---

# 👨‍💻 Author

## Hanan Ali Khan

**Software Engineering Student**

### Areas of Interest

- Artificial Intelligence
- Computer Vision
- Deep Learning
- Machine Learning
- Full-Stack Development
- Digital Forensics
- Computer Vision Research

---

### GitHub

https://github.com/Hananrana

---

# 📄 License

This project is licensed under the **MIT License**.

See the LICENSE file for more information.

---

# ⭐ Support the Project

If you found this repository useful,

⭐ Star the repository

🍴 Fork it

📢 Share it with others

Your support helps improve future AI research projects.

---

<div align="center">

# 🛡️ DeepShield AI

### Detecting AI-Generated Images & Deepfake Videos Using Artificial Intelligence

**Built with ❤️ using Python, PyTorch, Flask, OpenCV and Deep Learning**

---

### Thank you for visiting this repository!

⭐ Don't forget to Star the project ⭐

</div>
