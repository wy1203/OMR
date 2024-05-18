import psutil
from sklearn import svm
import cv2
import os
import random
import argparse
import numpy as np
import json

target_img_size = (32, 32)


def extract_hog_features(img):
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


def extract_features(img, feature_set='hog'):
    return extract_hog_features(img)


def get_directories(path):
    directories = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.bekrn') or file.endswith('.jpg'):
                directories.append(root)
                # break  # TODO: to be removed
    return directories


def pad_features(features, max_length):
    # Pad the feature vectors so they all have the same size
    padded_features = []
    for feature in features:
        if len(feature) < max_length:
            # If the feature is smaller than the max length, pad it
            padded_feature = np.pad(
                feature, (0, max_length - len(feature)), 'constant', constant_values=0)
        else:
            # If the feature is already max length, keep it as is
            padded_feature = feature
        padded_features.append(padded_feature)
    return padded_features


def get_available_memory():
    # Get available memory in bytes
    return psutil.virtual_memory().available


def estimate_memory_usage(num_samples, feature_size):
    # Estimate memory usage in bytes
    return num_samples * feature_size * np.dtype(np.float32).itemsize


# def load_features(path="./dataset", feature_set='hog', validation_split=0.2, max_memory_usage=4 * 1024**3):
#     labels = []
#     features = []
#     directories = get_directories(path)
#     random.seed(42)  # Ensure reproducibility
#     feature_size = None
#     for dir_name in directories:
#         img_filenames = [fn for fn in os.listdir(
#             dir_name) if fn.endswith('.jpg') or fn.endswith('.bekrn')]
#         # Pair each .bekrn file with its corresponding .jpg file
#         for fn in img_filenames:
#             if fn.endswith('.bekrn'):
#                 label = fn
#                 img_name = fn.replace('.bekrn', '.jpg')
#                 if img_name in img_filenames:  # Check if the corresponding .jpg file exists
#                     labels.append(label)

#                     img_path = os.path.join(dir_name, img_name)
#                     img = cv2.imread(img_path)
#                     feature = extract_features(img, feature_set)
#                     features.append(feature)
#                     # Set feature_size once we have the first feature vector
#                     if feature_size is None:
#                         feature_size = len(feature)

#                     current_memory_usage = estimate_memory_usage(
#                         len(features), feature_size)
#                     available_memory = get_available_memory()
#                     if current_memory_usage >= max_memory_usage or current_memory_usage >= available_memory:
#                         print(
#                             f"Stopping data loading to avoid memory overflow. Loaded {len(features)} samples.")
#                         break

#                 else:
#                     print(f"Image {img_name} not found for label {label}")

#         print('Finished processing: ', dir_name)

#     # Split the data into training and validation sets
#     combined = list(zip(features, labels))
#     random.shuffle(combined)
#     features[:], labels[:] = zip(*combined)

#     split_idx = int(len(features) * (1 - validation_split))
#     training_features, validation_features = features[:split_idx], features[split_idx:]
#     training_labels, validation_labels = labels[:split_idx], labels[split_idx:]

#     max_feature_length = max(len(f) for f in features)

#     # Pad all features to have the same length
#     training_features = pad_features(training_features, max_feature_length)
#     validation_features = pad_features(validation_features, max_feature_length)
#     # Create a mapping from unique string labels to integers
#     unique_labels = sorted(set(training_labels + validation_labels))
#     label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

#     # Convert the string labels to integers
#     training_labels = np.array([label_to_int[label]
#                                for label in training_labels], dtype=np.int32)
#     validation_labels = np.array([label_to_int[label]
#                                  for label in validation_labels], dtype=np.int32)

#     training_features = np.array(training_features)
#     # training_labels = np.array(training_labels, dtype=str)
#     validation_features = np.array(validation_features)
#     # validation_labels = np.array(validation_labels, dtype=str)


#     return training_features, training_labels, validation_features, validation_labels, label_to_int


def load_features(path="./dataset", feature_set='hog', max_memory_usage=4 * 1024 ** 3):
    train_features, train_labels = [], []
    val_features, val_labels = [], []
    test_features, test_labels = [], []

    train_count, val_count, test_count = 0, 0, 0
    max_train_count, max_val_count, max_test_count = 5000, 500, 500

    directories = get_directories(path)
    random.seed(42)  # Ensure reproducibility
    feature_size = None

    for dir_name in directories:
        img_filenames = [fn for fn in os.listdir(
            dir_name) if fn.endswith('.jpg') or fn.endswith('.bekrn')]
        for fn in img_filenames:
            if fn.endswith('.bekrn'):
                label = fn
                img_name = fn.replace('.bekrn', '.jpg')
                if img_name in img_filenames:
                    img_path = os.path.join(dir_name, img_name)
                    img = cv2.imread(img_path)
                    feature = extract_features(img, feature_set)

                    if feature_size is None:
                        feature_size = len(feature)

                    current_memory_usage = estimate_memory_usage(
                        len(train_features) + len(val_features) + len(test_features), feature_size)
                    available_memory = get_available_memory()
                    if current_memory_usage >= max_memory_usage or current_memory_usage >= available_memory:
                        print(
                            f"Stopping data loading to avoid memory overflow. Loaded {train_count + val_count + test_count} samples.")
                        break

                    if train_count < max_train_count:
                        train_features.append(feature)
                        train_labels.append(label)
                        train_count += 1
                    elif val_count < max_val_count:
                        val_features.append(feature)
                        val_labels.append(label)
                        val_count += 1
                    elif test_count < max_test_count:
                        test_features.append(feature)
                        test_labels.append(label)
                        test_count += 1

        print('Finished processing:', dir_name)
        if train_count >= max_train_count and val_count >= max_val_count and test_count >= max_test_count:
            break

    max_feature_length = max(len(f)
                             for f in train_features + val_features + test_features)

    # Pad all features to have the same length
    train_features = pad_features(train_features, max_feature_length)
    val_features = pad_features(val_features, max_feature_length)
    test_features = pad_features(test_features, max_feature_length)

    # Create a mapping from unique string labels to integers
    unique_labels = sorted(set(train_labels + val_labels + test_labels))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    # Save the label_to_int dictionary as a JSON file
    with open("label_to_int.json", "w") as f:
        json.dump(label_to_int, f)

    # Convert the string labels to integers
    train_labels = np.array([label_to_int[label]
                            for label in train_labels], dtype=np.int32)
    val_labels = np.array([label_to_int[label]
                          for label in val_labels], dtype=np.int32)
    test_labels = np.array([label_to_int[label]
                           for label in test_labels], dtype=np.int32)

    train_features = np.array(train_features)
    val_features = np.array(val_features)
    test_features = np.array(test_features)

    print(train_features.shape)
    print(val_features.shape)
    print(test_features.shape)

    return train_features, train_labels, val_features, val_labels, test_features, test_labels, label_to_int
