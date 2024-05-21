import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time

model = load_model('TestModel1.keras')
labels = ['backpain', 'chestpain', 'cough', 'falling_down', 'headache', 'neckpain']

def preprocess_image(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128))
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = frame.astype('float32') / 255.0
    return frame

def classify_pose(frame):
    processed_frame = preprocess_image(frame)
    prediction = model.predict(processed_frame)
    class_idx = np.argmax(prediction, axis=1)[0]
    return labels[class_idx]




st.title("Patient Monitoring System")
st.write("Webcam:")
run = st.checkbox('Run')
stframe = st.empty()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.stop()



while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Error: Could not read frame.")
        break

    pose = classify_pose(frame)
    stframe.image(frame, channels='BGR')
    st.write(f"Predicted Pose: {pose}")
    time.sleep(3.0)




cap.release()
