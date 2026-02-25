# 🤟 SIBI Real-Time Translator - AI Innovation
**By SMK Wikrama 1 Garut**

Aplikasi penerjemah Bahasa Isyarat SIBI (Sistem Isyarat Bahasa Indonesia) berbasis Kecerdasan Buatan (AI) yang mampu menerjemahkan gerakan tangan secara real-time menjadi teks dan suara (Text-to-Speech) dalam Bahasa Indonesia.

---

## 🌟 Fitur Utama
* **Real-Time Detection**: Menggunakan MediaPipe untuk pelacakan 21 titik landmark tangan dengan latensi rendah.
* **Modern Web Interface**: Antarmuka berbasis Streamlit yang user-friendly dan responsif.
* **Text-to-Speech (TTS)**: Mengeluarkan suara Bahasa Indonesia secara otomatis saat akurasi deteksi mencapai ambang batas tertentu.
* **Auto Data Collector**: Fitur perekaman data otomatis untuk mempercepat proses pembuatan dataset.
* **Mobile Accessibility**: Dapat diakses melalui HP teman dalam jaringan WiFi yang sama menggunakan fitur Local IP Server.

---

## 🛠️ Arsitektur Teknologi
* **Language**: Python 3.8+
* **Computer Vision**: MediaPipe, OpenCV
* **Deep Learning**: TensorFlow/Keras (Dense Neural Network)
* **Web Framework**: Streamlit
* **TTS Engine**: Pyttsx3

---

## 📂 Struktur Proyek
* `main.py` - Script untuk mengumpulkan dataset (Auto-capture).
* `train.py` - Script untuk melatih model Deep Learning.
* `predict.py` - Script untuk menjalankan translator versi desktop.
* `app.py` - Script utama untuk menjalankan translator versi Web Dashboard.
* `model.h5` - File model hasil training.
* `labels.txt` - Daftar label huruf yang dapat dideteksi.
* `requirements.txt` - Daftar library yang diperlukan.
* `run_translator.bat` - Script otomatis untuk menjalankan aplikasi di Windows.

---

## 🚀 Cara Penggunaan

### 1. Instalasi
Clone repository ini dan instal dependensi yang diperlukan:
```bash
git clone [https://github.com/username/sibi-translator.git](https://github.com/username/sibi-translator.git)
cd sibi-translator
pip install -r requirements.txt


Jalankan Aplikasi WEB
Gunakan perintah berikut untuk menjalankan antarmuka web:
streamlit run app.py


