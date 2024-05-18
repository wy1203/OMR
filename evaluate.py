import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, val_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features
            labels = labels

            outputs = model(features)
            # Get the class with the highest score
            predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()
            all_predictions.extend(predicted_labels.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    print(all_labels)
    print(all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1
