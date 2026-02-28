import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from dataset import PlantDataset
from model import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "../data/train"
VAL_DIR = "../data/val"

BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001


def train():
    print("Loading datasets...")

    train_dataset = PlantDataset(TRAIN_DIR)
    val_dataset = PlantDataset(VAL_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Number of classes: {len(train_dataset.classes)}")

    model = create_model(num_classes=len(train_dataset.classes)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("../models", exist_ok=True)
            torch.save(model.state_dict(), "../models/best_model.pth")

    print("Training complete.")


if __name__ == "__main__":
    train()
    