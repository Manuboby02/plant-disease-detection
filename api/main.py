from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn.functional as F
import sys
import os
import cv2
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from model import create_model

app = FastAPI(title="Plant Disease Detection API")

# -----------------------
# CONFIG
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pth"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "train"))

# Load class names
classes = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

# Image transform
transform = Compose([
    Resize(224, 224),
    Normalize(mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Load model once at startup
model = create_model(num_classes=len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


@app.get("/")
def home():
    return {"message": "Plant Disease Detection API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image"}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    augmented = transform(image=image_rgb)
    input_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 3)

    results = []

    for i in range(3):
        class_name = classes[top_indices[0][i].item()]
        confidence_score = top_probs[0][i].item() * 100

        clean_name = class_name.replace("___", " - ").replace("__", " ")

        results.append({
            "rank": i + 1,
            "prediction": clean_name,
            "confidence": round(confidence_score, 2)
        })

    return {"predictions": results}