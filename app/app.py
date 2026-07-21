from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

with open('../models/isl_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    coords = request.json['landmarks']
    pred = model.predict([coords])[0]
    label = le.inverse_transform([pred])[0]
    return jsonify({'prediction': label})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
