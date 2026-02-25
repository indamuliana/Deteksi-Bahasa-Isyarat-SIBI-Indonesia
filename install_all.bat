@echo off
title Installer SIBI Translator
echo Memulai instalasi library...

:: Pembuatan venv jika belum ada
if not exist venv (
    python -m venv venv
)

call venv\Scripts\activate

echo Menginstal dependencies dari requirements.txt...
pip install -r requirements.txt

echo Instalasi selesai!
pause