# Optical Music Recognition

CS4701 Group42

## Set Up

Clone the repository:

`git clone https://github.com/wy1203/OMR.git`

Install packages:

`pip install -r requirements.txt`

## How to Run 
- Load the dataset (see below for download instruction)
  ```
  cd $PATH_TO_OMR/OMR
  python extract_dataset.py
  ```
- Train and Evaluate:
  ```
  python train.py
  ```
- Demo (can only be run after training)
  ```
  python demo.py
  ```
  
## Introduction to the GrandStaff Dataset

- The dataset we use to train our model is from the paper [_End-to-end Optical Music Recognition for Pianoform Sheet Music_](https://link.springer.com/article/10.1007/s10032-023-00432-z), written by Ríos-Vila, A., Rizo, D., Iñesta, J.M. et al.
- The history of OMR process includes the following stages
  - Stage 1: \
    There is a first set in which the basic symbols such as note heads, beams, or accidentals (usually referred to as “primitives”) are detected
  - Stage 2: \
    heuristic strategies based on hand-crafted rules, such as the ones we read from Orchestra Project
- Scheme: `**bekern`, i.e. basic extended kern
  - based on [\*\*`kern` scheme](https://www.humdrum.org/guide/ch02/)
  - A `kern` file is a sequence of lines
  - Details please refer to [Hundrum Tool Kit](https://www.humdrum.org/rep/kern/)

## Convolutionary Recurrent Neural Network (CRNN)

- A hybrid neural network architecture that combines the feature extraction capabilities of CNNs with the sequence modeling capabilities of RNNs. This architecture is particularly suited for tasks that involve sequential input data with important spatial features, such as images or videos.
- Structure

  - _Convolutional Layers_: \
    The CRNN begins with a series of convolutional layers, which are responsible for extracting a hierarchy of spatial features from the input data. These layers automatically learn to identify patterns, textures, and various features that are crucial for understanding the content of the input.

  - _Recurrent Layers_: \
    The features extracted by the convolutional layers are then processed by recurrent layers. These layers, typically composed of Long Short-Term Memory (LSTM) units or Gated Recurrent Units (GRUs), are adept at capturing the temporal or sequential dependencies between the features.

  - _Output Layer_: \
    Finally, the output from the recurrent layers is passed through a dense layer (or layers) to make predictions. For sequence recognition tasks, the output is usually a sequence of symbols or characters.


