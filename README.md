# Child Safety Ecosystem - Backend API

Welcome to the **Child Safety Backend API**, the core intelligence engine powering the Child Safety Ecosystem. This FastAPI-based backend processes live video and audio streams, applies advanced AI models for threat detection, and provides an NLP-powered legal guidance system.

## 🚀 Key Modules and Features

This repository consists of several specialized modules developed across different branches, all contributing to a comprehensive child safety solution:

### 1. Nanny Cam Guardian & Video Streaming
*Branches: `thivina`, `integrated-to-frontend-(pp1)`*
- **Live Video Streamer**: Exposes endpoints and CORS configurations to stream real-time video directly to the frontend React Native dashboard.
- **Person Tracking & Pose Estimation**: Utilizes YOLO to track individuals, improving the accuracy of classifying children vs. adults in the frame.
- **Physical Threat Detection**: Analyzes body poses and interactions to detect potentially abusive or dangerous physical incidents in real time.

### 2. Advanced Face Recognition & Hazard Detection
*Branch: `thivina-known-people&trained-knife-model`*
- **Known-People Recognition Tooling**: Integrates facial recognition capabilities to identify registered family members (parents/nannies) and alert on unrecognized individuals.
- **Fine-Tuned Hazard Models**: Includes a specialized, fine-tuned knife detection model to accurately identify environmental hazards.

### 3. Audio Guardian (Acoustic Alerts)
*Branch: `vidusha-backend-acustic-alert`*
- **Audio Processing**: Listens to and processes incoming audio streams from the monitoring environment.
- **Threat Classification**: Detects acoustic anomalies such as crying, screaming, or harsh tones, generating real-time alerts for the Guardian Dashboard.

### 4. Legal RAG System
*Branch: `Malithi`*
- **Retrieval-Augmented Generation (RAG)**: A dedicated `legal-rag-backend` that queries a comprehensive penal code database (`penal.json`).
- **AI Legal Guidance**: Provides immediate, context-aware legal advice and next steps in cases of suspected child abuse, directly accessible from the frontend app.

## 🛠️ Technology Stack
- **Framework**: FastAPI (Python)
- **Computer Vision**: OpenCV, YOLO (Ultralytics), MediaPipe
- **Audio Processing**: Librosa / Audio classification models
- **NLP**: Retrieval-Augmented Generation architectures

## 🏃 Getting Started

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

*(Note: Ensure you have your `.env` variables and necessary model weights properly configured before running.)*
