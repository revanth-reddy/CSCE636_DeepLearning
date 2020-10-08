import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype = np.float32)
        y_train: An numpy array of shape [50000, ].
            (dtype = np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype = np.float32)
        y_test: An numpy array of shape [10000, ].
            (dtype = np.int32)
    """

    ### YOUR CODE HERE
    num_train_batches = 5
    img_pixels = 32

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding = 'bytes')
        return dict

    def get_data(_filepath):
        data = unpickle(_filepath)

        raw_img_float = np.array(data[b'data'], dtype = float) / 255.0
        img = raw_img_float.reshape([-1, 3, img_pixels, img_pixels])
        img = img.transpose([0, 2, 3, 1])
        lbl = np.array(data[b'labels'])
        return img, lbl

    images = np.zeros(shape = [0, img_pixels, img_pixels, 3], dtype = float)
    labels = np.asarray([], dtype = int)

    # For each data-file.
    for i in range(num_train_batches):
        filename = "data_batch_" + str(i + 1)
        filepath = os.path.join(data_dir, filename)
        images_batch, labels_batch = get_data(filepath)
        images = np.concatenate((images, images_batch))
        labels = np.concatenate((labels, labels_batch))

    x_train, y_train = images, labels

    filename = "test_batch"
    filepath = os.path.join(data_dir, filename)
    x_test, y_test = get_data(filepath)

    ### END CODE HERE
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    return x_train, y_train, x_test, y_test

def train_valid_split(x_train, y_train, split_index = 45000):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000, ].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index, ].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index, ].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid

