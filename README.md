# Garbage Identify & Classification 🚮

A YOLOv8-based Garbage Identification and Classification system that detects and classifies roadside garbage into multiple categories using Deep Learning.  
The project also includes a Flask-based web application for real-time image upload and prediction.

---

## 📌 Project Overview

This project is designed to:
- Detect garbage objects from road images
- Classify them into predefined categories
- Provide real-time results via a web interface

The system is suitable for **smart city**, **road monitoring**, and **waste management** applications.

---

## 🧠 Model Details

- **Algorithm**: YOLOv8 (Object Detection)
- **Framework**: Ultralytics YOLO
- **Training Platform**: Google Colab (NVIDIA T4 GPU)
- **Backend**: Python Flask
- **Frontend**: HTML + JavaScript

---

## 🗂️ Garbage Categories

The model classifies garbage into **6 categories**:
1. BIODEGRADABLE  
2. CARDBOARD  
3. GLASS  
4. METAL  
5. PAPER  
6. PLASTIC  

---

## 📊 Model Performance (Final)

- **Precision**: ~61%
- **Recall**: ~48%
- **mAP@50**: ~55%
- **Inference Speed**: Real-time capable

> Dataset imbalance exists (e.g., fewer PAPER samples), which affects recall.

---
