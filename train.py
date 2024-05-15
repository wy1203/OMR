import os
from extract_features import load_features
from model import build_crnn_model, initialize_model_params
from evaluate import evaluate_model
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

def save_model(model, model_path='model.h5'):
    model.save(model_path)
    print(f"Model saved to {model_path}")

def plot_metrics(history):
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

def train():
    print("Loading Training Data")
    (train_features, train_labels, train_input_length, train_label_length), (val_features, val_labels, val_input_length, val_label_length) = load_features()
    print("Training Data loaded")
    print()

    print("Initializing Model")
    # Adjust input shape according to the actual feature shape
    input_shape = (train_features.shape[1],)
    params = initialize_model_params(input_shape, 100)
    model = build_crnn_model(params)
    print("Model Initialized")

    # Compile the model
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    print("Model Compiled")

    print()
    print("Start Training")
    history = model.fit(
        [train_features, train_labels, train_input_length, train_label_length],
        train_labels,
        epochs=10,
        validation_data=([val_features, val_labels, val_input_length, val_label_length], val_labels)
    )
    print("Training Completed")

    print()
    print("Evaluating Model")
    evaluate_model(model, val_features, val_labels)
    print("Evaluation Completed")

    save_model(model)
    plot_metrics(history)

train()