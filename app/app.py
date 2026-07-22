import gradio as gr
import mediapipe as mp
import pickle
import spaces

with open('isl_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

@spaces.GPU
def predict(frame):
    if frame is None:
        return "No image received"
    result = hands.process(frame)
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        pred = model.predict([coords])[0]
        conf = max(model.predict_proba([coords])[0])
        label = le.inverse_transform([pred])[0]
        return f"{label}  ({round(conf*100)}% confidence)"
    return "No hand detected — reposition and try again"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(sources=["webcam"], type="numpy"),
    outputs=gr.Textbox(label="Detected Sign"),
    title="ISL Sign Detector"
)
demo.launch()
