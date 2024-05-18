import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import CRNN, initialize_model_params, ctc_loss_func
from evaluate import evaluate_model
import matplotlib.pyplot as plt
import numpy as np
from extract_features import load_features

# Create dataset and dataloaders


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def save_model(model, model_path='model.pth'):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def plot_metrics(train_losses, val_losses):
    plt.figure(figsize=(12, 4))

    # Plot training & validation loss values
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()


def train():
    print("Loading Training Data")
    train_features, train_labels, val_features, val_labels, test_features, test_labels, label_to_int = load_features()
    print("Training Data loaded")
    print()

    print("Initializing Model")
    input_dim = train_features.shape[1]
    vocabulary_size = len(label_to_int)  # example vocabulary size
    params = initialize_model_params(input_dim, vocabulary_size)
    model = CRNN(params).to(device)
    print("Model Initialized")

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = CustomDataset(train_features, train_labels)
    val_dataset = CustomDataset(val_features, val_labels)
    test_dataset = CustomDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()  # Use Cross-Entropy Loss

    print("Start Training")
    num_epochs = 10
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            labels = labels.to(dtype=torch.long)
            # Reshape outputs to (batch_size, num_classes)

            loss = criterion(outputs, labels)  # Calculate Cross-Entropy Loss
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)

                outputs = model(features)
                labels = labels.to(dtype=torch.long)
                # Calculate Cross-Entropy Loss
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()

        val_losses.append(epoch_val_loss / len(val_loader))
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}")

    print("Training Completed")

    print()
    print("Evaluating Model")
    evaluate_model(model, test_loader)
    print("Evaluation Completed")

    save_model(model)
    plot_metrics(train_losses, val_losses)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train()
