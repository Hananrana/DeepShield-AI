# DeepShield — Enterprise Deepfake Detection Platform

DeepShield is a production-ready, enterprise-grade deepfake detection platform designed to verify digital authenticity in real-time. Built with PyTorch, OpenCV, Flask, and MySQL, DeepShield leverages both spatial textures (RGB pixel manipulations) and spectral frequency analytics (Fast Fourier Transform) to distinguish authentic faces from synthetic media with high precision.

---

## 📂 Project Directory Structure

```text
DeepShield/
│
├── .env.example                  # Template file for environment configurations
├── requirements.txt              # Project Python dependency specifications
│
├── app.py                        # Main Flask server entry point (Routing, API, & Logic)
├── db.py                         # MySQL connection pool setup & database query utilities
├── migrate.py                    # Database creation, schema migrations, & seeding tool
│
├── inference/                    # Machine learning models & prediction pipelines
│   ├── image_predict.py          # Image deepfake prediction pipeline (RGB + FFT Gated Fusion)
│   ├── video_predict.py          # Video deepfake prediction pipeline (Xception + LSTM + Temporal Attention)
│   └── test_env.py               # Utility to verify local PyTorch, CUDA, and GPU environment
│
├── models/                       # Pre-trained deep learning weight files (*.pth)
│   ├── fusion_rgb_fft_best_v2.pth   # Weights for the dual-branch image Gated Fusion model
│   └── phase4_rgb_only_clean_best.pth # Weights for the Xception-LSTM temporal video model
│
├── static/                       # Static web resources
│   ├── style.css                 # Main application stylesheets (Custom UI & Animations)
│   ├── bg.png                    # Dashboard homepage background graphic
│   │
│   ├── js/
│   │   └── main.js               # Frontend interaction, drag-and-drop, & live charts
│   │
│   ├── images/                   # UI graphics, dashboard icons, and sample demo images
│   │   ├── authentic_face.png
│   │   ├── authentic_fft.png
│   │   ├── authentic_gradcam.png
│   │   ├── robot_scanner.png
│   │   └── ... (Other static graphics & user flow assets)
│   │
│   ├── uploads/                  # Temporary directory for user-uploaded media files
│   └── results/                  # Directory for generated forensic artifacts (heatmaps, etc.)
│
├── templates/                    # Jinja2 HTML layout views
│   ├── index.html                # Main interactive detection dashboard
│   ├── login.html                # Enterprise login portal
│   ├── signup.html               # Secure new user enrollment portal
│   ├── verify_otp.html           # Multi-factor email OTP validation screen
│   ├── forgot_password.html      # Account recovery initiation screen
│   ├── reset_password.html       # Account recovery target password reset screen
│   ├── change_password.html      # Authorized profile password update view
│   ├── admin_dashboard.html      # Administrator security audit & user management portal
│   ├── saas_security.html        # SaaS product feature overview section
│   ├── privacy.html              # Privacy policy agreement template
│   └── terms.html                # Terms of service agreement template
│
├── test_image_model.py           # Command-line tool to load and print image model parameters
└── test_prediction.py            # Local command-line tool to run image prediction test case
```

---

## 🔍 Detailed File & Module Breakdown

### 1. Root Configuration & Application Files
*   **[app.py](file:///c:/Users/hanan/Desktop/DeepShield/DeepShield/DeepShield/app.py)**: The central orchestrator of the web platform. It initializes the Flask application, loads environment variables, configures OAuth (Google & GitHub login), implements JWT token authentication (including session rotation), controls routing for all templates, handles file uploads, runs predictions on images or videos, and routes AI chatbot conversations via the Google Gemini API.
*   **[db.py](file:///c:/Users/hanan/Desktop/DeepShield/DeepShield/DeepShield/db.py)**: Establishes a robust MySQL connection pool (`MySQLConnectionPool`) using configuration variables defined in the `.env` file. Provides safe, parameterized helper functions to execute SQL statements (`execute_query`) to protect the system against SQL Injection.
*   **[migrate.py](file:///c:/Users/hanan/Desktop/DeepShield/DeepShield/DeepShield/migrate.py)**: Sets up the MySQL database environment from scratch. It creates the required schema and tables (`users`, `otp_verifications`, `password_resets`, `user_sessions`, `login_history`, `security_logs`), inserts the default administrator credentials, and migrates existing user data from a local SQLite database (`deepshield.db`) if found.
*   **[requirements.txt](file:///c:/Users/hanan/Desktop/DeepShield/DeepShield/DeepShield/requirements.txt)**: Specifies package requirements for the project including PyTorch (`torch`, `torchvision`), machine learning backbones (`timm`), image processing (`opencv-python`, `pillow`), authentication helper tools (`bcrypt`, `pyjwt`, `authlib`), database connectivity (`mysql-connector-python`), and utility packages (`python-dotenv`, `resend`, `requests`).
*   **[test_image_model.py](file:///c:/Users/hanan/Desktop/DeepShield/DeepShield/DeepShield/test_image_model.py)**: A diagnostic script to verify that the image prediction model loads correctly, prints its detection threshold, and calculates the total number of trainable model parameters.
*   **[test_prediction.py](file:///c:/Users/hanan/Desktop/DeepShield/DeepShield/DeepShield/test_prediction.py)**: A script that runs an end-to-end inference pass on a local test image (`test.jpg`) to output the classification label and confidence score.

### 2. Machine Learning Inference Pipeline (`inference/`)
*   **[image_predict.py](file:///c:/Users/hanan/Desktop/DeepShield/DeepShield/DeepShield/inference/image_predict.py)**:
    *   Uses MTCNN (`facenet-pytorch`) to locate and extract faces from uploaded images.
    *   Generates 2D Fast Fourier Transforms (FFT) of faces to analyze frequency anomalies caused by generative model upsampling.
    *   Defines the **GatedFusionHead** class: A gated attention neural layer that dynamically weights spatial RGB texture channels and frequency-domain FFT representations.
    *   Defines the **FusionModel** class: A dual-branch architecture extracting features using an EfficientNet backbone.
*   **[video_predict.py](file:///c:/Users/hanan/Desktop/DeepShield/DeepShield/DeepShield/inference/video_predict.py)**:
    *   Segments videos into temporal sequence arrays (defaulting to 16 keyframes).
    *   Defines an **RGBDetector** using an Xception spatial backbone coupled to a Bi-directional LSTM sequence layer.
    *   Implements **TemporalAttention**: A custom attention head that assigns dynamic importance weights to individual video frames, highlighting temporal inconsistencies standard in frame-swapping deepfakes.
*   **[test_env.py](file:///c:/Users/hanan/Desktop/DeepShield/DeepShield/DeepShield/inference/test_env.py)**: A diagnostic script reporting the installed versions of Torch, Torchvision, OpenCV, NumPy, and checking if CUDA-accelerated GPU inference is available.

### 3. Front-End Templates & Resources (`templates/` & `static/`)
*   **[index.html](file:///c:/Users/hanan/Desktop/DeepShield/DeepShield/DeepShield/templates/index.html)**: The primary dashboard interface. Contains the interactive dropzone for drag-and-drop file upload, visualizations for the confidence score (dynamic circular gauges), real-time progress timelines, FFT heatmaps, and a custom Gemini-powered chat assistant.
*   **[login.html](file:///c:/Users/hanan/Desktop/DeepShield/DeepShield/DeepShield/templates/login.html)** / **[signup.html](file:///c:/Users/hanan/Desktop/DeepShield/DeepShield/DeepShield/templates/signup.html)**: User authentication interfaces. Employs front-end validation, password strength bars (requiring enterprise-grade complexity), and hooks for Google/GitHub single sign-on (SSO).
*   **[verify_otp.html](file:///c:/Users/hanan/Desktop/DeepShield/DeepShield/DeepShield/templates/verify_otp.html)**: Provides a clean PIN input UI to verify the 6-digit Multi-Factor Authentication (MFA) OTP codes dispatched to users upon signing up or logging in.
*   **[admin_dashboard.html](file:///c:/Users/hanan/Desktop/DeepShield/DeepShield/DeepShield/templates/admin_dashboard.html)**: Restricted administration view mapping registered platform users, security logs (XSS attempts, logins, OTP failures), and system usage history with pagination and search tools.
*   **[main.js](file:///c:/Users/hanan/Desktop/DeepShield/DeepShield/DeepShield/static/js/main.js)**: Handles asynchronous front-end routines. Orchestrates chunked AJAX media uploads, formats UI responses, feeds dynamic metrics into real-time visual charts, and powers the chat container interactions.
*   **[style.css](file:///c:/Users/hanan/Desktop/DeepShield/DeepShield/DeepShield/static/style.css)**: Implements custom stylesheets with premium glassmorphism layouts, subtle neon borders, glow transitions, and responsive grid layouts.

---

## 🔒 Security Architecture Highlights

DeepShield integrates enterprise security practices directly into its routing and request lifecycles:
1.  **Authentication & Sessions**: Implements a double-cookie JWT pattern (access token expires in 15 min, refresh token in 7 days). Session persistence is cross-checked against database records (`user_sessions`).
2.  **CSRF Defense**: Verifies a unique double-submitted CSRF token via HTTP headers (`X-CSRF-Token`) or form data for all POST, PUT, DELETE, and PATCH methods.
3.  **Security Headers & CSP**: Enforces strict headers, including `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, and a restrictive Content Security Policy (CSP) blocking unauthorized external scripts and CDNs.
4.  **Rate Limiting**: Protects authentication endpoints (login, signup, OTP, password recovery) with an in-memory rate-limiter blocking abuse patterns.
5.  **Data Sanitization**: Employs XSS filtering (`sanitize_string`) and strict password complexity rules.

---

## 🚀 Setup & Execution Guide

### 1. Configure the Environment
Duplicate `.env.example` to a new file named `.env` and fill in your custom keys:
```bash
# Database Configuration
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=deepshield

# Flask Secret Keys
SECRET_KEY=generate_a_secure_long_random_string_here
JWT_SECRET_KEY=generate_another_secure_jwt_key_here

# Resend API for OTP dispatch
RESEND_API_KEY=re_your_api_key
RESEND_FROM_EMAIL=onboarding@resend.dev

# Google Gemini API key for the dashboard chatbot assistant
GEMINI_API_KEY=your_gemini_api_key

# OAuth 2.0 Credentials (Optional)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Database Migrations
Initialize the MySQL schema, create the tables, and seed the system admin:
```bash
python migrate.py
```
> [!NOTE]
> During development, if no mail dispatcher is active, generated verification OTP codes are saved to static/uploads/otp_debug.json for testing convenience.

### 4. Boot the Application
Launch the Flask development server:
```bash
python app.py
```
The platform will run locally at `http://127.0.0.1:5000`. You can log in using the seeded administrator credentials:
- **Email**: `admin@deepshield.ai`
- **Password**: `AdminPassword123!`
