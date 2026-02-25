import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import queue
from collections import deque
from typing import List, Any

# Import MediaPipe
from mediapipe.python.solutions.hands import Hands, HAND_CONNECTIONS
from mediapipe.python.solutions import drawing_utils as mp_drawing

# ==========================
# 1. Konfigurasi TTS (Bahasa Indonesia & Thread-Safe)
# ==========================
class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.engine.setProperty('volume', 1.0)
        
        # Pengecekan tipe eksplisit untuk menghindari Pylance Error
        voices = self.engine.getProperty('voices')
        if isinstance(voices, list):
            for voice in voices:
                v_lang = getattr(voice, 'languages', [])
                v_name = getattr(voice, 'name', "").lower()
                # Prioritaskan suara Bahasa Indonesia
                if "id" in v_lang or "indonesia" in v_name:
                    self.engine.setProperty('voice', voice.id)
                    break
        
        self.last_spoken = ""
        self.msg_queue = queue.Queue()
        # Worker thread agar kamera tidak macet saat bicara
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            text = self.msg_queue.get()
            if text is not None:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except:
                    pass
            self.msg_queue.task_done()

    def speak(self, text):
        if text != self.last_spoken and text != "Mendeteksi...":
            self.last_spoken = text
            self.msg_queue.put(text)

tts = TextToSpeech()

# ==========================
# 2. Load Model & Labels
# ==========================
model = tf.keras.models.load_model("model.h5")

with open("labels.txt", "r") as f:
    labels: List[str] = [line.strip() for line in f.readlines()]

# ==========================
# 3. Fungsi Helper
# ==========================
def normalize_landmarks(landmark_list: List[float]) -> List[float]:
    if len(landmark_list) != 63:
        return []
    base_x, base_y, base_z = landmark_list[0], landmark_list[1], landmark_list[2]
    normalized = []
    for i in range(0, len(landmark_list), 3):
        normalized.append(landmark_list[i] - base_x)
        normalized.append(landmark_list[i + 1] - base_y)
        normalized.append(landmark_list[i + 2] - base_z)
    return normalized

# ==========================
# 4. MediaPipe Setup
# ==========================
hands = Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==========================
# 5. Konfigurasi
# ==========================
prediction_queue = deque(maxlen=10)
UI_CONFIDENCE_THRESHOLD = 0.80     
VOICE_CONFIDENCE_THRESHOLD = 0.95  

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results: Any = hands.process(rgb)

    predicted_label = "Mendeteksi..."
    current_confidence = 0.0
    confidence_text = "-"

    if results and getattr(results, "multi_hand_landmarks", None):
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, list(HAND_CONNECTIONS))

            landmarks_raw: List[float] = []
            for lm in hand_landmarks.landmark:
                landmarks_raw.extend([lm.x, lm.y, lm.z])

            if len(landmarks_raw) == 63:
                normalized_input = normalize_landmarks(landmarks_raw)
                input_data = np.array(normalized_input, dtype=np.float32).reshape(1, -1)
                
                prediction = model.predict(input_data, verbose=0)
                current_confidence = float(np.max(prediction))
                class_id = int(np.argmax(prediction))

                if current_confidence > UI_CONFIDENCE_THRESHOLD:
                    prediction_queue.append(class_id)
                    if len(prediction_queue) > 0:
                        final_class = max(set(prediction_queue), key=prediction_queue.count)
                        predicted_label = labels[final_class]
                        confidence_text = f"{current_confidence * 100:.1f}%"

                if current_confidence >= VOICE_CONFIDENCE_THRESHOLD:
                    tts.speak(predicted_label)
    else:
        # Reset memori suara jika tangan hilang
        tts.last_spoken = ""

    # ==========================
    # 6. UI Overlay (Modern Design)
    # ==========================
    overlay = frame.copy()
    theme_color = (0, 200, 0) if current_confidence >= VOICE_CONFIDENCE_THRESHOLD else (200, 0, 0)
    
    # Header & Footer Semi-Transparan
    cv2.rectangle(overlay, (0, 0), (w, 115), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, h-40), (w, h), (10, 10, 10), -1)
    
    # Apply Transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Progress Bar Akurasi
    bar_width = int(w * current_confidence)
    cv2.rectangle(frame, (0, 112), (bar_width, 115), theme_color, -1)

    # Teks Informasi Utama
    cv2.putText(frame, "HASIL TERJEMAHAN:", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    cv2.putText(frame, predicted_label, (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)

    # Indikator Status Suara
    if current_confidence >= VOICE_CONFIDENCE_THRESHOLD:
        cv2.circle(frame, (w-40, 45), 10, (0, 255, 0), -1)
        cv2.putText(frame, "VOICE", (w-85, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Teks Kredit SMK Wikrama 1 Garut
    credit_text = "SIBI Translator | SMK Wikrama 1 Garut"
    cv2.putText(frame, credit_text, (20, h-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    cv2.putText(frame, "ESC to Exit", (w-100, h-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    cv2.imshow("SIBI Translator - SMK Wikrama 1 Garut", frame)

    # Tombol ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()