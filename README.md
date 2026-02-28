# 🌿 Plant Disease Detection System

An end-to-end deep learning system for multi-class plant disease classification built using PyTorch and EfficientNet-B0.

This project includes:
- Streamlit web interface for interactive predictions
- FastAPI REST backend for production-style inference
- Transfer learning for efficient model training

---

## 🚀 Project Overview

This system classifies plant leaf images into 15 disease categories using transfer learning with EfficientNet-B0 pretrained on ImageNet.

The objective was not only to achieve high accuracy but also to build a complete machine learning pipeline including:

- Data preprocessing
- Model training and evaluation
- Web UI integration
- REST API backend service

---

## 🧠 Model Details

- Architecture: EfficientNet-B0
- Input Size: 224x224
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Framework: PyTorch

### 📊 Performance

Controlled Test Dataset:
- 99% Accuracy
- Strong precision and recall across all classes

Real-World Images:
- ~75–85% accuracy
- Performance drop due to domain shift, lighting variation, and background noise

---

## 🏗️ System Architecture

User (Streamlit UI)
        ↓
FastAPI Backend
        ↓
PyTorch Model
        ↓
Prediction (Top-3 with confidence)

The model can be accessed in two ways:

1. Streamlit UI (human interface)
2. FastAPI REST API (machine-to-machine interface)

---

## 🔗 API Endpoints

### GET /
Returns API health status.

### POST /predict
Accepts an image file and returns JSON predictions:

{
  "predictions": [
    {
      "rank": 1,
      "prediction": "Tomato - Early blight",
      "confidence": 97.23
    }
  ]
}

Interactive API documentation is available at:
http://127.0.0.1:8000/docs

---

## 📁 Project Structure

plant-disease-project/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── app/
│   └── streamlit_app.py
│
├── api/
│   └── main.py
│
├── split_dataset.py
├── requirements.txt
└── README.md

---

## 🛠️ Tech Stack

- Python
- PyTorch
- EfficientNet (timm)
- Albumentations
- OpenCV
- Streamlit
- FastAPI
- Uvicorn

---

## 🚀 How To Run

1. Create virtual environment:

python -m venv venv  
venv\Scripts\activate  

2. Install dependencies:

pip install -r requirements.txt  

3. Run Streamlit UI:

streamlit run app/streamlit_app.py  

4. Run FastAPI backend:

uvicorn api.main:app --reload  

---

## 🔍 Key Learnings

- Transfer learning accelerates model development
- High dataset accuracy does not guarantee real-world robustness
- Separation of UI and backend improves scalability
- REST APIs enable ML model deployment

---

## 👤 Author

Manu Boby  
M.Tech ML & AI