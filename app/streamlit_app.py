import streamlit as st
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


# --------------------
# CONFIG
# --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pth"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "train"))


# --------------------
# Load Class Names
# --------------------
classes = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])


# --------------------
# Load Model
# --------------------
@st.cache_resource
def load_model():
    model = create_model(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()


# --------------------
# Image Transform
# --------------------
transform = Compose([
    Resize(224, 224),
    Normalize(mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# --------------------
# Streamlit UI
# --------------------
st.title("🌿 Plant Disease Detection System")

st.write("Upload a leaf image and the model will predict the disease.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Uploaded Image", width="stretch")

    augmented = transform(image=image_rgb)
    input_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)

        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities, 3)

    st.subheader("🔍 Top Predictions:")

    for i in range(3):
        class_name = classes[top_indices[0][i].item()]
        confidence_score = top_probs[0][i].item() * 100

        # Clean class name formatting
        clean_name = class_name.replace("___", " - ").replace("__", " ")

        st.write(f"{i+1}. **{clean_name}** — {confidence_score:.2f}%")