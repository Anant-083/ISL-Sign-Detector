"""
app/app.py

Flask backend for the ISL Sign Detector.

Flow per request:
  1. Frontend sends a webcam frame (base64 image) to /predict
  2. Extract hand landmarks (MediaPipe)
  3. Run through trained model (static classifier or LSTM, once trained)
  4. Apply confidence threshold + temporal smoothing
  5. Return confirmed word (if any) to frontend
  6. Frontend accumulates words -> calls /construct_sentence -> Groq -> gTTS

NOTE: Model loading is stubbed until models/isl_model.h5 (or .pkl) exists
from the training phase. Placeholder logic is marked with TODO.
"""

import os
import base64
import io

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv
from gtts import gTTS

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.preprocess import HandLandmarkExtractor, SequenceBuffer, flatten_hand
from utils.smoothing import PredictionSmoother, WordBuffer

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---- Global objects (loaded once at startup) ----
extractor = HandLandmarkExtractor()
seq_buffer = SequenceBuffer(sequence_length=30, feature_dim=63)
smoother = PredictionSmoother(window_size=10, confidence_threshold=0.80)
word_buffer = WordBuffer()

# TODO: Once training is done, load your real model here.
# Example for a Random Forest (static signs):
#   import joblib
#   static_model = joblib.load("models/static_model.pkl")
#
# Example for an LSTM (dynamic signs):
#   from tensorflow.keras.models import load_model
#   dynamic_model = load_model("models/isl_model.h5")
static_model = None
dynamic_model = None

# TODO: Replace with your real label list once you finalize your word set
LABELS = ["HELLO", "THANK_YOU", "YES", "NO", "PLEASE"]


def decode_base64_image(base64_string):
    """Converts a base64-encoded image (from frontend canvas) into an OpenCV BGR frame."""
    img_data = base64.b64decode(base64_string.split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives a single frame, returns a confirmed word if the smoothing
    window agrees on a stable prediction.
    """
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    frame = decode_base64_image(data["image"])
    hands = extractor.extract_from_frame(frame)

    if not hands:
        return jsonify({"prediction": None, "confidence": 0.0})

    flat = flatten_hand(hands[0])

    # --- Static sign path ---
    if static_model is not None:
        # TODO: adapt to your model's actual predict/predict_proba interface
        probs = static_model.predict_proba([flat])[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        predicted_label = LABELS[pred_idx]

        confirmed = smoother.update(predicted_label, confidence)
        if confirmed:
            word_buffer.add_word(confirmed)

        return jsonify({
            "prediction": predicted_label,
            "confidence": confidence,
            "confirmed": confirmed,
            "sentence_so_far": word_buffer.get_sentence_raw(),
        })

    # --- Dynamic sign path (sequence-based) ---
    seq_buffer.add_frame(flat)
    if seq_buffer.is_ready() and dynamic_model is not None:
        sequence = np.expand_dims(seq_buffer.get_sequence(), axis=0)  # (1, 30, 63)
        probs = dynamic_model.predict(sequence, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        predicted_label = LABELS[pred_idx]

        confirmed = smoother.update(predicted_label, confidence)
        if confirmed:
            word_buffer.add_word(confirmed)

        return jsonify({
            "prediction": predicted_label,
            "confidence": confidence,
            "confirmed": confirmed,
            "sentence_so_far": word_buffer.get_sentence_raw(),
        })

    # No model loaded yet -- still return landmark detection success
    return jsonify({"prediction": None, "confidence": 0.0, "note": "Model not loaded yet"})


@app.route("/construct_sentence", methods=["POST"])
def construct_sentence():
    """
    Takes the raw accumulated word buffer and asks Groq to turn it
    into a grammatically correct sentence.
    """
    raw_words = word_buffer.get_sentence_raw()
    if not raw_words:
        return jsonify({"sentence": ""})

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        prompt = (
            f"Convert these ISL sign words into a natural, grammatically correct "
            f"English sentence. Words: {raw_words}. "
            f"Respond with ONLY the sentence, nothing else."
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        sentence = response.choices[0].message.content.strip()
    except Exception as e:
        sentence = raw_words  # fallback: just show raw words if Groq fails
        print(f"Groq error: {e}")

    return jsonify({"sentence": sentence})


@app.route("/speak", methods=["POST"])
def speak():
    """Converts text to speech and returns an audio file."""
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    tts = gTTS(text=text, lang="en")
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    return send_file(audio_buffer, mimetype="audio/mpeg")


@app.route("/reset", methods=["POST"])
def reset():
    """Clears the word buffer and smoothing history (e.g. new sentence)."""
    smoother.reset()
    seq_buffer.reset()
    word_buffer.clear()
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)