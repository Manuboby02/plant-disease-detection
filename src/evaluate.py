import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

from dataset import PlantDataset
from model import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DIR = "../data/test"
MODEL_PATH = "../models/best_model.pth"
BATCH_SIZE = 32


def evaluate():
    print("Loading test dataset...")

    test_dataset = PlantDataset(TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = create_model(num_classes=len(test_dataset.classes)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    evaluate()