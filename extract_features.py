from sklearn import svm
import cv2
import os
import random
import argparse
import numpy as np

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
                break
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


def load_features(feature_set='hog'):
    labels = []
    features = []
    directories = get_directories('./dataset')

    for dir_name in directories:
        img_filenames = [fn for fn in os.listdir(
            dir_name) if fn.endswith('.jpg') or fn.endswith('.bekrn')]
        # Pair each .bekrn file with its corresponding .png file
        for fn in img_filenames:
            if fn.endswith('.bekrn'):
                label = fn
                img_name = fn.replace('.bekrn', '.jpg')
                if img_name in img_filenames:  # Check if the corresponding .png file exists
                    labels.append(label)

                    img_path = os.path.join(dir_name, img_name)
                    img = cv2.imread(img_path)
                    features.append(extract_features(img, feature_set))
                else:
                    print(f"Image {img_name} not found for label {label}")
        break
        print('Finished processing: ', dir_name)
    max_feature_length = max(len(f) for f in features)
    # Pad all features to have the same length
    features = pad_features(features, max_feature_length)
    # features n x len(h), labels n
    return features, labels
