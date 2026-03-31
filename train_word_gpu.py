import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np

from model_one_word import Net
from dataset_one_word import VideoDataset


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total

    print("Prediction distribution:", Counter(all_preds))

    return acc


def main():

    print("SCRIPT STARTED")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    DATA_PATH = r"D:/grid_word_dataset/"

    # -------------------------
    # Use ALL 34 speakers
    # -------------------------
    speakers = [f"s{i}" for i in range(1, 35)]

    dataset = VideoDataset(DATA_PATH, speakers)

    print("Total dataset size:", len(dataset))

    # Print class distribution
    label_list = [label for label, _ in dataset.pathList]
    print("Class distribution:", Counter(label_list))

    # -------------------------
    # Train / Validation Split
    # -------------------------
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print("Train size:", len(train_dataset))
    print("Validation size:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    # -------------------------
    # Model
    # -------------------------
    model = Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # 🔥 Increased learning rate
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 30 

    for epoch in range(EPOCHS):

        model.train()
        correct = 0
        total = 0
        total_loss = 0

        for videos, labels in train_loader:

            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(videos)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total

        val_acc = evaluate(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {total_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

    torch.save(model.state_dict(), "word_model_gpu.pth")
    print("Training complete. Model saved as word_model_gpu.pth")


if __name__ == "__main__":
    main()
