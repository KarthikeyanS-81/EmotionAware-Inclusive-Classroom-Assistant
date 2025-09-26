from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from modules.emotion_detection.model import load_emotion_model, predict_emotion

app = Flask(__name__)
CORS(app)

# Load the pre-trained emotion detection model
emotion_model = load_emotion_model()

@app.route('/api/emotion', methods=['POST'])
def analyze_emotion():
    # Expecting a frame in base64 or as a file upload
    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    emotion = predict_emotion(frame, emotion_model)
    return jsonify({'emotion': emotion})

@app.route('/')
def home():
    return "Emotion Aware Classroom Backend Running!"

if __name__ == '__main__':
    app.run(debug=True)