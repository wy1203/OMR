import torch
import gradio as gr
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
import numpy as np
import psutil
import cv2
import json

# Define the model architecture (same as the one used during training)


class CRNN(torch.nn.Module):
    def __init__(self, params):
        super(CRNN, self).__init__()

        self.rnn_layers = params['rnn_layers']
        self.rnn_units = params['rnn_units']
        self.input_dim = params['input_dim']
        self.vocabulary_size = params['vocabulary_size']

        self.linear = nn.Linear(self.input_dim, 2 * self.rnn_units)
        self.rnn = nn.ModuleList()
        for _ in range(self.rnn_layers):
            self.rnn.append(nn.LSTM(2 * self.rnn_units, self.rnn_units,
                                    bidirectional=True, batch_first=True))
            self.rnn.append(nn.Dropout(0.25))

        self.output = nn.Linear(self.rnn_units * 2, self.vocabulary_size)

    def forward(self, x):
        x = self.linear(x)
        for layer in self.rnn:
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            else:
                x = layer(x)
        x = self.output(x)
        return x


# Parameters used to initialize the model (these should match your training parameters)
params = {
    'input_dim': 1764,  # example input dimension
    'rnn_units': 512,  # example number of RNN units
    'rnn_layers': 2,   # example number of RNN layers
    'vocabulary_size': 6  # 6 classes for the 6 composers
}

# Load the model
model = CRNN(params)
model.load_state_dict(torch.load(
    'model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the class names
class_names = ["Ludwig van Beethoven", "Frédéric Chopin", "Johann Nepomuk Hummel",
               "Scott Joplin", "Wolfgang Amadeus Mozart", "Domenico Scarlatti"]

# Function to extract HOG features from the image


def extract_hog_features(img):
    target_img_size = (32, 32)
    img = cv2.resize(img, target_img_size)
    win_size = (32, 32)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)
    block_size = (block_size_in_cells[1] * cell_size[1],
                  block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size,
                            block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h


def classify_image(image):
    if image is None:
        raise gr.Error("No image uploaded")

    image = Image.open(image).convert("L")  # Convert to grayscale
    image = np.array(image)
    hog_features = extract_hog_features(image)
    hog_features = torch.tensor(hog_features).float(
    ).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(hog_features)
        _, predicted = torch.max(outputs, 1)
        class_name = class_names[predicted.item()]

    return class_name


demo = gr.Blocks()

image_classifier = gr.Interface(
    fn=classify_image,
    inputs=[
        gr.inputs.Image(type="filepath",
                        label="Upload Image of Musical Score"),
    ],
    outputs="text",
    layout="horizontal",
    title="Composer Classification from Musical Score",
    description=(
        "Music Score Recognition. By Janna Lin (jnl77), Aishwarya Velu (av382), and Yao Wu (yw582). CS 4701 Group 42. Currently supports recognition for Ludwig van Beethoven, Frédéric Chopin, Johann Nepomuk Hummel, Scott Joplin, Wolfgang Amadeus Mozart, and Domenico Scarlatti."
    ),
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([image_classifier], ["Image Classifier"])

demo.launch(enable_queue=True, share=True)
