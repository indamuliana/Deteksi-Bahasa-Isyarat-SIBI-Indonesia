import cv2
import os
import csv
import time
from typing import List
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

# ==============================
# 1. Setup Folder
# ==============================

DATASET_PATH = "dataset"

if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# ==============================
# 2. Normalisasi Landmark
# ==============================

def normalize_landmarks(landmark_list: List[float]) -> List[float]:
    """
    Normalisasi relatif terhadap wrist (landmark 0)
    """
    base_x, base_y, base_z = landmark_list[0], landmark_list[1], landmark_list[2]

    normalized = []
    for i in range(0, len(landmark_list), 3):
        normalized.append(landmark_list[i] - base_x)
        normalized.append(landmark_list[i + 1] - base_y)
        normalized.append(landmark_list[i + 2] - base_z)

    return normalized

# ==============================
# 3. MediaPipe Setup
# ==============================

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Ubah frozenset menjadi list agar tidak error di Pylance
HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

# ==============================
# 4. Camera Setup
# ==============================

cap = cv2.VideoCapture(0)

print("--- SIBI DATA COLLECTOR READY ---")
print("1. Pastikan tangan terlihat landmark.")
print("2. Tekan tombol A-Z untuk simpan 1 sampel.")
print("3. Tekan ESC untuk keluar.")

last_save_time = 0
SAVE_DELAY = 0.4  # Anti spam

# ==============================
# 5. Main Loop
# ==============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    landmark_list: List[float] = []

    multi_hand_landmarks = getattr(results, "multi_hand_landmarks", None)

    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                HAND_CONNECTIONS
            )

            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

    # UI Overlay
    cv2.rectangle(frame, (0, 0), (400, 50), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "TEKAN A-Z UNTUK SIMPAN DATA",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        1
    )

    cv2.imshow("Dataset Collector - SIBI", frame)

    key = cv2.waitKey(1) & 0xFF

    # ==============================
    # 6. Save Logic (Protected)
    # ==============================

    if multi_hand_landmarks and len(landmark_list) == 63:
        if ord('a') <= key <= ord('z'):

            current_time = time.time()

            if current_time - last_save_time > SAVE_DELAY:
                label = chr(key).upper()
                file_path = os.path.join(DATASET_PATH, f"{label}.csv")

                data_to_save = normalize_landmarks(landmark_list)

                with open(file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(data_to_save)

                # Hitung total sampel
                with open(file_path, "r") as f:
                    total_samples = sum(1 for _ in f)

                print(f"[SUCCESS] {label} total sampel: {total_samples}")

                last_save_time = current_time

    # ESC to exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()