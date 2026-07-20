# ISL Sign Detector

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-00897B?logo=google&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy&logoColor=white)
![Google Colab](https://img.shields.io/badge/Trained%20on-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

A real-time Indian Sign Language (ISL) fingerspelling recognizer that uses hand landmark detection and a machine learning classifier to identify static ISL alphabet signs from a live webcam feed, deployed as a web app.

## Overview

This project detects hand shapes via webcam and classifies them into ISL alphabet letters (A–Z, excluding H, J, Y which require motion). Instead of processing raw video frames through a heavy CNN, it uses **MediaPipe Hands** to extract 21 key hand landmark points (fingertips, knuckles, wrist) per frame, then feeds those coordinates into a lightweight **Random Forest classifier** to predict the corresponding letter. This landmark-based approach keeps the model fast, small, and deployable on free-tier cloud hosting.

**Live demo:** https://isl-sign-detector.onrender.com

## Features

- Real-time webcam-based sign detection running entirely in the browser
- Lightweight ML pipeline (landmarks, not raw images) — fast inference, low resource usage
- Mobile-responsive UI with front/back camera switch support
- Fully deployed on Render (free tier)

## Tech Stack

| Layer | Technology |
|---|---|
| Hand landmark detection | MediaPipe Hands |
| Classifier | scikit-learn (Random Forest) |
| Backend | Flask |
| Frontend | HTML/CSS/JavaScript (browser camera capture via `getUserMedia`) |
| Training environment | Google Colab (T4 GPU) |
| Deployment | Render |
| Dataset | Indian Sign Language Dataset (Kaggle) |

## How It Works

1. **Capture** — Browser accesses the webcam via JavaScript and captures frames periodically.
2. **Landmark extraction** — Each frame is sent to the Flask backend, where MediaPipe detects 21 (x, y, z) hand landmark points.
3. **Prediction** — The 63-value landmark vector (21 points × 3 coordinates) is passed to a trained Random Forest model, which predicts the corresponding letter.
4. **Display** — The predicted letter is sent back to the browser and shown in real time.

## Project Structure
```
ISL-Sign-Detector/
├── app/
│ ├── app.py # Flask app: routes, model loading, prediction logic
│ ├── templates/
│ │ └── index.html # Frontend UI
│ └── static/
│ └── style.css # Styling
├── models/
│ ├── isl_model.pkl # Trained Random Forest classifier
│ └── label_encoder.pkl # Label encoder (maps predictions to letters)
├── data/landmarks/ # (Reserved for extracted landmark datasets)
├── utils/
│ ├── preprocess.py # Landmark extraction & normalization utilities
│ └── smoothing.py # Prediction smoothing / word buffering utilities
├── requirements.txt
└── README.md
```
## Model Details

- **Training data:** ~500 labeled hand images across 23 static ISL alphabet letters
- **Input features:** 63 values per sample (21 landmarks × x, y, z coordinates)
- **Algorithm:** Random Forest Classifier (200 estimators)
- **Test accuracy:** ~71%

## Known Limitations

- Trained on a small dataset (21–46 images per letter), so accuracy varies with lighting, hand angle, and skin tone differences from the training data
- Only supports static fingerspelling signs — motion-based letters (H, J, Y) and full ISL vocabulary/grammar are out of scope
- Best performance requires good lighting and the full hand clearly visible in frame

## Local Setup

```bash
git clone https://github.com/Anant-083/ISL-Sign-Detector.git
cd ISL-Sign-Detector
pip install -r requirements.txt
cd app
python app.py
```

Open `http://localhost:5000` in your browser and allow camera access.

## Deployment

Deployed on Render using:
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `cd app && gunicorn app:app --timeout 120 --workers 1`

## Author

**Anant Paul** — B.Tech CSE (AI & ML), Brainware University
[GitHub](https://github.com/Anant-083) · [LinkedIn](https://linkedin.com/in/anant-paul-5852a333b/)
