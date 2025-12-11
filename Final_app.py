import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import random
import time

# -----------------------------
# Custom CSS Styles
# -----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(90deg, rgba(255,107,107,0.1) 0%, rgba(255,154,0,0.1) 100%);
    }
    
    .stRadio > div {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .stButton > button {
        background-color: #4ECDC4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #3AADA9;
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.1);
    }
    
    .calc-display {
        background-color: #F1F8FF;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .prediction-box {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4ECDC4;
    }
    
    .success-box {
        background-color: rgba(78, 205, 196, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4ECDC4;
    }
    
    .warning-box {
        background-color: rgba(255, 154, 0, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #FF9A00;
    }
    
    .error-box {
        background-color: rgba(255, 107, 107, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #FF6B6B;
    }
    
    .camera-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .calculation-result {
        font-size: 1.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin: 1rem 0;
    }
    
    .mode-selector {
        margin-bottom: 2rem;
    }
    
    .upload-container {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = 'gesture_calculator_model.h5'
IMG_SIZE = 64
model = load_model(MODEL_PATH)

# Class names (digits + operators)
class_names = ['0','1','2','3','4','5','6','7','8','9','+','/','*','-']
operator_map = {'s': '-', 'm': '*', 'a': '+', 'd': '/'}

# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(image):
    """Predict class from uploaded image or camera frame."""
    if isinstance(image, Image.Image):
        img = np.array(image.convert('RGB'))
    else:  # OpenCV frame
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    preds = model.predict(img)
    idx = np.argmax(preds)
    predicted_class = class_names[idx]
    confidence = preds[0][idx]
    
    return predicted_class, confidence

# -----------------------------
# Streamlit UI
# -----------------------------
st.markdown('<h1 class="main-header">🖐 Sign Language Calculator </h1>', unsafe_allow_html=True)

mode = st.radio("Select Mode", ["Upload Image", "Live Camera"], key="mode_selector")

# Session state
if "first_number" not in st.session_state:
    st.session_state.first_number = ""
if "operator" not in st.session_state:
    st.session_state.operator = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "running" not in st.session_state:
    st.session_state.running = False

# -----------------------------
# Mode 1: Upload Image
# -----------------------------
if mode == "Upload Image":
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Gesture Image", type=['jpg','jpeg','png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        predicted_class, confidence = predict_image(image)
        
        st.markdown(f'''
        <div class="prediction-box">
            <h3>Predicted Gesture: <strong>{predicted_class}</strong> ({confidence*100:.2f}%)</h3>
        </div>
        ''', unsafe_allow_html=True)

        # Map custom operators
        if predicted_class in operator_map:
            predicted_class = operator_map[predicted_class]

        # Handle numbers
        if predicted_class.isdigit():
            if st.session_state.operator == "":
                st.session_state.first_number += predicted_class
                st.markdown(f'''
                <div class="calc-display">
                    <h3>First Number: {st.session_state.first_number}</h3>
                </div>
                ''', unsafe_allow_html=True)
            else:
                second_number = predicted_class
                st.markdown(f'''
                <div class="calc-display">
                    <h3>Second Number: {second_number}</h3>
                </div>
                ''', unsafe_allow_html=True)

                # Calculate
                try:
                    if st.session_state.operator == '+':
                        st.session_state.result = int(st.session_state.first_number) + int(second_number)
                    elif st.session_state.operator == '-':
                        st.session_state.result = int(st.session_state.first_number) - int(second_number)
                    elif st.session_state.operator == '*':
                        st.session_state.result = int(st.session_state.first_number) * int(second_number)
                    elif st.session_state.operator == '/':
                        st.session_state.result = int(st.session_state.first_number) / int(second_number)

                    st.markdown(f'''
                    <div class="calculation-result">
                        Result: {st.session_state.result}
                    </div>
                    ''', unsafe_allow_html=True)

                    # Reset for next calculation
                    st.session_state.first_number = str(st.session_state.result)
                    st.session_state.operator = ""
                except Exception as e:
                    st.markdown(f'''
                    <div class="error-box">
                        <h3>Calculation error: {e}</h3>
                    </div>
                    ''', unsafe_allow_html=True)
                    st.session_state.first_number = ""
                    st.session_state.operator = ""
                    st.session_state.result = None

        # Handle operators
        elif predicted_class in ['+', '-', '*', '/']:
            st.session_state.operator = predicted_class
            st.markdown(f'''
            <div class="calc-display">
                <h3>Operator set to: {st.session_state.operator}</h3>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="warning-box">
                <h3>Unknown gesture detected!</h3>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Reset Calculator"):
        st.session_state.first_number = ""
        st.session_state.operator = ""
        st.session_state.result = None
        st.markdown('''
        <div class="success-box">
            <h3>Calculator Reset!</h3>
        </div>
        ''', unsafe_allow_html=True)

# -----------------------------
# Mode 2: Live Camera
# -----------------------------
elif mode == "Live Camera":
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ Start Camera & Calculator")
    with col2:
        stop_btn = st.button("⛔ Stop")

    camera_placeholder = st.empty()
    calc_placeholder = st.empty()

    operators = ['+', '-', '*', '/']

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False
        st.markdown('''
        <div class="warning-box">
            <h3>Stopped camera!</h3>
        </div>
        ''', unsafe_allow_html=True)

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        st.markdown('''
        <div class="success-box">
            <h3>Camera started! Calculation every 3 seconds...</h3>
        </div>
        ''', unsafe_allow_html=True)

        last_calc_time = time.time()

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.markdown('''
                <div class="error-box">
                    <h3>Failed to access camera!</h3>
                </div>
                ''', unsafe_allow_html=True)
                break

            camera_placeholder.image(frame, channels="BGR")

            # Generate random calculation every 3 seconds
            if time.time() - last_calc_time > 3:
                st.session_state.first_number = random.randint(0, 9)
                st.session_state.second_number = random.randint(0, 9)
                st.session_state.operator = random.choice(operators)

                # Calculate result
                try:
                    if st.session_state.operator == '+':
                        st.session_state.result = st.session_state.first_number + st.session_state.second_number
                    elif st.session_state.operator == '-':
                        st.session_state.result = st.session_state.first_number - st.session_state.second_number
                    elif st.session_state.operator == '*':
                        st.session_state.result = st.session_state.first_number * st.session_state.second_number
                    elif st.session_state.operator == '/':
                        st.session_state.result = round(st.session_state.first_number / st.session_state.second_number, 2)
                except Exception:
                    st.session_state.result = "Error"

                calc_placeholder.markdown(f'''
                <div class="calc-display">
                    <h3>🎲 Calculation: {st.session_state.first_number} {st.session_state.operator} {st.session_state.second_number} = {st.session_state.result}</h3>
                </div>
                ''', unsafe_allow_html=True)

                last_calc_time = time.time()

            time.sleep(0.1)

        cap.release()