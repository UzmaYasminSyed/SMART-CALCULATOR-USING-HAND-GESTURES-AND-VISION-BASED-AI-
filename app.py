import streamlit as st
import cv2

st.title("Smart Calculator Using Hand Gestures")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()

    if not ret:
        st.write("Camera stopped")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

camera.release()
