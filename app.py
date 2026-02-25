import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import queue
import socket
from collections import deque
from typing import List, Any

# Import MediaPipe
from mediapipe.python.solutions.hands import Hands, HAND_CONNECTIONS
from mediapipe.python.solutions import drawing_utils as mp_drawing

# ==========================
# 1. Fungsi Cek IP Lokal
# ==========================
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# ==========================
# 2. Konfigurasi TTS
# ==========================
class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        
        voices = self.engine.getProperty('voices')
        if isinstance(voices, list):
            for voice in voices:
                v_lang = getattr(voice, 'languages', [])
                v_name = getattr(voice, 'name', "").lower()
                if "id" in v_lang or "indonesia" in v_name:
                    self.engine.setProperty('voice', voice.id)
                    break
        
        self.last_spoken = ""
        self.msg_queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            text = self.msg_queue.get()
            if text:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except:
                    pass
            self.msg_queue.task_done()

    def speak(self, text):
        if text != self.last_spoken:
            self.last_spoken = text
            self.msg_queue.put(text)

# ==========================
# 3. Load Resource & Cache
# ==========================
@st.cache_resource
def get_resources():
    model = tf.keras.models.load_model("model.h5")
    hands_obj = Hands(
        static_image_mode=False, 
        max_num_hands=1, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    with open("labels.txt", "r") as f:
        labels_list = [line.strip() for line in f.readlines()]
    tts_obj = TextToSpeech()
    return model, hands_obj, labels_list, tts_obj

model, hands, labels, tts = get_resources()

# ==========================
# 4. Helper: Normalisasi
# ==========================
def normalize_landmarks(landmark_list):
    base_x, base_y, base_z = landmark_list[0], landmark_list[1], landmark_list[2]
    normalized = []
    for i in range(0, len(landmark_list), 3):
        normalized.append(landmark_list[i] - base_x)
        normalized.append(landmark_list[i+1] - base_y)
        normalized.append(landmark_list[i+2] - base_z)
    return normalized

# ==========================
# 5. UI Layout
# ==========================
st.set_page_config(page_title="SIBI Translator - Wikrama", layout="wide")

local_ip = get_local_ip()
st.sidebar.title("Koneksi Perangkat")
st.sidebar.info(f"Akses dari HP:\nhttp://{local_ip}:8501")

st.title("🤟 App Pendeteksi Bahasa Isyarat - SIBI Indonesia")
st.caption("AI Application by SMK Wikrama 1 Garut")

col1, col2 = st.columns([2, 1])

with col1:
    st.write("### Live Camera Feed")
    FRAME_WINDOW = st.image([])
    run_app = st.checkbox("Nyalakan Kamera", value=True)

with col2:
    st.write("### Hasil Terjemahan")
    result_placeholder = st.empty()
    st.write("---")
    conf_placeholder = st.empty()

# ==========================
# 6. Loop Pemrosesan
# ==========================
cap = cv2.VideoCapture(0)
prediction_queue = deque(maxlen=10)

while run_app:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Gunakan Any untuk menghindari Pylance Error pada tipe data MediaPipe
    results: Any = hands.process(rgb_frame)
    
    predicted_label = "Mendeteksi..."
    confidence_val = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                list(HAND_CONNECTIONS)
            )
            
            landmarks_raw = []
            for lm in hand_landmarks.landmark:
                landmarks_raw.extend([lm.x, lm.y, lm.z])
            
            if len(landmarks_raw) == 63:
                norm_data = normalize_landmarks(landmarks_raw)
                input_data = np.array(norm_data, dtype=np.float32).reshape(1, -1)
                
                prediction = model.predict(input_data, verbose=0)
                confidence_val = float(np.max(prediction))
                
                if confidence_val > 0.80:
                    class_id = int(np.argmax(prediction))
                    prediction_queue.append(class_id)
                    
                    if len(prediction_queue) > 0:
                        final_class = max(set(prediction_queue), key=prediction_queue.count)
                        predicted_label = labels[final_class]
                        
                        if confidence_val >= 0.95:
                            tts.speak(predicted_label)
    else:
        tts.last_spoken = ""

    # Render Visual ke Web
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    result_placeholder.markdown(f"""
        <div style="background-color:#1e3a8a; padding:30px; border-radius:15px; text-align:center;">
            <p style="color:#cbd5e1; margin:0; font-size:14px;">TERDETEKSI:</p>
            <h1 style="color:white; font-size:120px; margin:0;">{predicted_label}</h1>
        </div>
    """, unsafe_allow_html=True)
    
    conf_placeholder.metric("Confidence Score", f"{confidence_val*100:.1f}%")

cap.release()