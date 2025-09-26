import cv2
import numpy as np
from tensorflow.keras.models import load_model

def load_emotion_model():
    # For demo, loads a pre-trained model (replace with your own)
    return load_model('backend/modules/emotion_detection/emotion_model.h5')

def predict_emotion(frame, model):
    # TODO: Improve preprocessing for your use case
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48)).reshape(1, 48, 48, 1) / 255.0
    preds = model.predict(face)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotion_labels[int(np.argmax(preds))]