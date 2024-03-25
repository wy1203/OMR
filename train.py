"""
Training the classifier
"""

from extract_features import load_features
from model import build_crnn_model, initialize_model_params


def train():
    print("Loading Training Data")
    train_features, train_labels = load_features()
    print("Traning Data loaded")
    print()

    print("Initializing Model")
    params = initialize_model_params(128, 256, 1, 100)
    model = build_crnn_model(params)
    print("Model Initialized")

    print()
    print("Start Training")
    model.fit(train_features, train_labels)
    print("Training Completed")


train()
