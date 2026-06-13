  import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# --------------------------
# 1. Dataset Preparation
# --------------------------
class CTScanDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob(os.path.join(image_dir, "*/*.png"))  # assuming PNG scans
        self.labels = [1 if 'tumor' in p else 0 for p in self.image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)  # for grayscale channel
        label = self.labels[idx]

        if self.transform:
            img = self.transform(torch.tensor(img, dtype=torch.float32))

        return img / 255.0, torch.tensor(label, dtype=torch.float32)

# --------------------------
# 2. Model Definition
# --------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*56*56, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32*56*56)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# --------------------------
# 3. Training Function
# --------------------------
def train_model(model, dataloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.view(-1,1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")
    print("Training complete")

# --------------------------
# 4. Evaluation and Tumor Size Estimation
# --------------------------
def evaluate(model, image_dir):
    model.eval()
    confidence_scores = []
    tumor_sizes = []
    image_paths = glob(os.path.join(image_dir, "*.png"))
    
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (224,224))
        input_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)/255.0
        with torch.no_grad():
            confidence = model(input_tensor).item()
            confidence_scores.append(confidence)
        
        # Estimate tumor size in pixels (simple thresholding)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        tumor_size = np.sum(thresh > 0)
        tumor_sizes.append(tumor_size)
    
    return confidence_scores, tumor_sizes

# --------------------------
# 5. Visualization
# --------------------------
def plot_results(confidences, sizes):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    plt.title("Confidence Scores")
    plt.xlabel("Confidence")
    plt.ylabel("Count")

    plt.subplot(1,2,2)
    plt.hist(sizes, bins=20, color='salmon', edgecolor='black')
    plt.title("Tumor Sizes (pixels)")
    plt.xlabel("Size (pixels)")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.show()

# --------------------------
# 6. Main Execution
# --------------------------
if __name__ == "__main__":
    # Data preparation
    transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
    train_dataset = CTScanDataset("data/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Model
    model = SimpleCNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    train_model(model, train_loader, criterion, optimizer, epochs=5)

    # Evaluation
    confidences, sizes = evaluate(model, "data/test")

    # Visualization
    plot_results(confidences, sizes)