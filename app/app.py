from flask import Flask, render_template, request, jsonify
import mediapipe as mp
import pickle
import numpy as np
import base64
import cv2

app = Flask(__name__)

with open('../models/isl_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label = ""
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        pred = model.predict([coords])[0]
        label = le.inverse_transform([pred])[0]

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
