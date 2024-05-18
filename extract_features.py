import psutil
from sklearn import svm
import cv2
import os
import random
import argparse
import numpy as np
import json

# Global composer to integer mapping
COMPOSER_TO_INT = {
    "beethoven": 0,
    "chopin": 1,
    "hummel": 2,
    "joplin": 3,
    "mozart": 4,
    "scarlatti-d": 5
}

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


def load_features(path="./dataset", feature_set='hog', max_memory_usage=4 * 1024 ** 3):
    train_features, train_labels = [], []
    val_features, val_labels = [], []
    test_features, test_labels = [], []

    # Initialize train, validation, and test counts
    train_count = {composer: 0 for composer in COMPOSER_TO_INT}
    val_count = {composer: 0 for composer in COMPOSER_TO_INT}
    test_count = {composer: 0 for composer in COMPOSER_TO_INT}
    max_train_count, max_val_count, max_test_count = 3000, 100, 100

    directories = get_directories(path)
    random.seed(42)  # Ensure reproducibility
    feature_size = None

    for dir_name in directories:
        # Extract composer name from directory path
        composer_name = dir_name.split(os.sep)[-3]
        if composer_name not in COMPOSER_TO_INT:
            print(f"Unknown composer {composer_name} found, skipping.")
            continue

        # Skip composer if data for this composer has reached the maximum count
        if train_count[composer_name] >= max_train_count and \
           val_count[composer_name] >= max_val_count and \
           test_count[composer_name] >= max_test_count:
            continue

        img_filenames = [fn for fn in os.listdir(
            dir_name) if fn.endswith('.jpg') or fn.endswith('.bekrn')]
        for fn in img_filenames:
            if fn.endswith('.bekrn'):
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
                            f"Stopping data loading to avoid memory overflow. Loaded {len(train_features) + len(val_features) + len(test_features)} samples.")
                        break

                    label = COMPOSER_TO_INT[composer_name]

                    if train_count[composer_name] < max_train_count:
                        train_features.append(feature)
                        train_labels.append(label)
                        train_count[composer_name] += 1
                    elif val_count[composer_name] < max_val_count:
                        val_features.append(feature)
                        val_labels.append(label)
                        val_count[composer_name] += 1
                    elif test_count[composer_name] < max_test_count:
                        test_features.append(feature)
                        test_labels.append(label)
                        test_count[composer_name] += 1

        print('Finished processing:', dir_name)
        if all(count >= max_train_count for count in train_count.values()) and \
           all(count >= max_val_count for count in val_count.values()) and \
           all(count >= max_test_count for count in test_count.values()):
            break

    max_feature_length = max(len(f)
                             for f in train_features + val_features + test_features)

    # Pad all features to have the same length
    train_features = pad_features(train_features, max_feature_length)
    val_features = pad_features(val_features, max_feature_length)
    test_features = pad_features(test_features, max_feature_length)

    # Save the COMPOSER_TO_INT dictionary as a JSON file
    with open("composer_to_int.json", "w") as f:
        json.dump(COMPOSER_TO_INT, f)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels, dtype=np.int32)
    val_features = np.array(val_features)
    val_labels = np.array(val_labels, dtype=np.int32)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels, dtype=np.int32)

    print(train_features.shape)
    print(val_features.shape)
    print(test_features.shape)

    return train_features, train_labels, val_features, val_labels, test_features, test_labels, COMPOSER_TO_INT
