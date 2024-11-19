import cv2
from ultralytics import YOLOWorld
import streamlit as st
from PIL import Image
import numpy as np

# Load the trained YOLO model
model = YOLOWorld('./best.pt')

st.title("YOLO Object Detection - Webcam")
run = st.checkbox("Run Webcam")

# Webcam feed
if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        stframe = st.empty()  # Streamlit's video placeholder
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam disconnected.")
                break

            # YOLO model inference
            results = model(frame)
            annotated_frame = results[0].plot()

            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)

            # Display on Streamlit
            stframe.image(img)

        cap.release()
