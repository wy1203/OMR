import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, val_features, val_labels):
    predictions = model.predict(val_features)
    predicted_labels = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(val_labels, predicted_labels)
    precision = precision_score(val_labels, predicted_labels, average='macro')
    recall = recall_score(val_labels, predicted_labels, average='macro')
    f1 = f1_score(val_labels, predicted_labels, average='macro')

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1