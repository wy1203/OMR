import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import CRNN, initialize_model_params
from evaluate import evaluate_model
import matplotlib.pyplot as plt
import numpy as np
from extract_features import load_features
import itertools

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


def train(train_features, train_labels, val_features, val_labels, test_features, test_labels, num_epochs, learning_rate, batch_size, label_to_int):
    input_dim = train_features.shape[1]
    vocabulary_size = len(label_to_int)  # example vocabulary size
    params = initialize_model_params(input_dim, vocabulary_size)
    model = CRNN(params).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_dataset = CustomDataset(train_features, train_labels)
    val_dataset = CustomDataset(val_features, val_labels)
    test_dataset = CustomDataset(test_features, test_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()  # Use Cross-Entropy Loss

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
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()

        val_losses.append(epoch_val_loss / len(val_loader))

    print(
        f"Training Completed for num_epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}")

    return model, train_losses, val_losses


def grid_search():
    # Define hyperparameters to search over
    num_epochs_list = [10, 20, 30, 50, 100]
    learning_rate_list = [0.001, 0.005, 0.0001, 0.0005, 0.00001]
    batch_size_list = [16, 32, 64, 128, 256]

    print("Loading Training Data")
    train_features, train_labels, val_features, val_labels, test_features, test_labels, label_to_int = load_features()
    print("Training Data loaded")

    results = []

    for num_epochs, learning_rate, batch_size in itertools.product(num_epochs_list, learning_rate_list, batch_size_list):
        model, train_losses, val_losses = train(
            train_features, train_labels, val_features, val_labels, test_features, test_labels, num_epochs, learning_rate, batch_size, label_to_int)

        # Evaluate the model
        model.eval()
        test_loader = DataLoader(CustomDataset(
            test_features, test_labels), batch_size=batch_size, shuffle=False)
        accuracy = evaluate_model(model, test_loader)

        results.append((num_epochs, learning_rate, batch_size, accuracy))

    # Output the results in a table format
    print("\nGrid Search Results:")
    print("Epochs | Learning Rate | Batch Size | Accuracy")
    for result in results:
        print(
            f"{result[0]:<6} | {result[1]:<13} | {result[2]:<10} | {result[3]:.4f}")

    # Optionally, save the results to a file
    with open("grid_search_results.txt", "w") as f:
        f.write("Epochs | Learning Rate | Batch Size | Accuracy\n")
        for result in results:
            f.write(
                f"{result[0]:<6} | {result[1]:<13} | {result[2]:<10} | {result[3]:.4f}\n")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid_search()