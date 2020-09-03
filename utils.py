import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Function to load data
def pre_process_nn_data(all_data):

    # Split data into training set, validation set and testing set, with ratio 6: 2: 2
    train_data, valid_and_test_data = train_test_split(all_data, test_size=0.4, random_state=0)
    valid_data, test_data = train_test_split(valid_and_test_data, test_size=0.5, random_state=0)

    # Select features and labels of train data
    x_train = train_data.iloc[:, 3:].values
    y_train = train_data.iloc[:, 2].values
    # Select features and labels of train data
    x_valid = valid_data.iloc[:, 3:].values
    y_valid = valid_data.iloc[:, 2].values
    # Select features and labels of train data
    x_test = test_data.iloc[:, 3:].values
    y_test = test_data.iloc[:, 2].values

    # Scale data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.astype(np.float32))
    x_valid_scaled = scaler.transform(x_valid.astype(np.float32))
    x_test_scaled = scaler.transform(x_test.astype(np.float32))

    # PCA to reduce dimensionality
    pca = PCA(5)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_valid_pca = pca.transform(x_valid_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    print('Train data: x', x_train_pca.shape, 'y', y_train.shape)
    print('Valid data: x', x_valid_pca.shape, 'y', y_valid.shape)
    print('Test data: x', x_test_pca.shape, 'y', y_test.shape)

    return (x_train_pca, y_train), (x_valid_pca, y_valid), (x_test_pca, y_test), scaler, pca


# Function to load data
def pre_process_cnn_data(patch_size, all_data):

    # Split data into training set, validation set and testing set, with ratio 6: 2: 2
    train_data, valid_and_test_data = train_test_split(all_data, test_size=0.4, random_state=0)
    valid_data, test_data = train_test_split(valid_and_test_data, test_size=0.5, random_state=0)

    # Select features and labels of train data
    x_train = train_data.iloc[:, 3:].values
    y_train = train_data.iloc[:, 2].values
    # Select features and labels of train data
    x_valid = valid_data.iloc[:, 3:].values
    y_valid = valid_data.iloc[:, 2].values
    # Select features and labels of train data
    x_test = test_data.iloc[:, 3:].values
    y_test = test_data.iloc[:, 2].values

    # Scale data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.astype(np.float32)).reshape(-1, 23, patch_size, patch_size).transpose((0, 2, 3, 1))
    x_valid_scaled = scaler.transform(x_valid.astype(np.float32)).reshape(-1, 23, patch_size, patch_size).transpose((0, 2, 3, 1))
    x_test_scaled = scaler.transform(x_test.astype(np.float32)).reshape(-1, 23, patch_size, patch_size).transpose((0, 2, 3, 1))

    # PCA to reduce dimensionality
    pca = PCA(5)
    x_train_pca = pca.fit_transform(x_train_scaled.reshape(-1, 23)).reshape(-1, patch_size, patch_size, 5)
    x_valid_pca = pca.transform(x_valid_scaled.reshape(-1, 23)).reshape(-1, patch_size, patch_size, 5)
    x_test_pca = pca.transform(x_test_scaled.reshape(-1, 23)).reshape(-1, patch_size, patch_size, 5)

    print('Train data: x', x_train_pca.shape, 'y', y_train.shape)
    print('Valid data: x', x_valid_pca.shape, 'y', y_valid.shape)
    print('Test data: x', x_test_pca.shape, 'y', y_test.shape)

    return (x_train_pca, y_train), (x_valid_pca, y_valid), (x_test_pca, y_test), scaler, pca


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 2)
    plt.show()
