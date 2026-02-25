import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras import layers, models, callbacks, utils

# ==============================
# 1. Konfigurasi Path
# ==============================
DATASET_PATH = "dataset"
MODEL_NAME = "model.h5"
LABEL_NAME = "labels.txt"

X = []
y = []

print("Membaca dataset...")

for file in os.listdir(DATASET_PATH):
    if file.endswith(".csv"):
        label = file.replace(".csv", "")
        file_path = os.path.join(DATASET_PATH, file)

        if os.path.getsize(file_path) == 0:
            continue

        try:
            data = pd.read_csv(file_path, header=None, on_bad_lines="skip")
            print(f"Memuat {file}: {len(data)} sampel.")

            # Validasi jumlah kolom
            if data.shape[1] != 63:
                print(f"Peringatan: {file} memiliki jumlah kolom tidak valid ({data.shape[1]})")
                continue

            for row in data.values:
                if len(row) == 63:
                    X.append(row)
                    y.append(label)

        except Exception as e:
            print(f"Gagal membaca {file}: {e}")

if len(X) == 0:
    print("Dataset masih kosong. Silakan ambil data dulu via main.py")
    exit()

X = np.array(X, dtype=np.float32)
y = np.array(y)

# ==============================
# 2. Encoding Label
# ==============================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = utils.to_categorical(y_encoded)
num_classes = len(label_encoder.classes_)

with open(LABEL_NAME, "w") as f:
    for label in label_encoder.classes_:
        f.write(label + "\n")

# ==============================
# 3. Split Data
# ==============================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ==============================
# 4. Arsitektur Model
# ==============================
model = models.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(32, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# 5. Training
# ==============================
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

print("\nMemulai training...")

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose="auto"
)

model.save(MODEL_NAME)

print(f"\nSelesai! Model disimpan di {MODEL_NAME}")