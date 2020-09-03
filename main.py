import numpy as np
import pandas as pd
import os

from models.Normal_CNN import train_normal_cnn
from models.Dilated_CNN import train_dilated_cnn
from models.Basic_network import train_basic_nn


# basic nn
def test_basic_nn(patch_size, data_path, save_dir):
    # Read data
    all_data = pd.read_csv(data_path, header=None)

    accuracies = []
    f_scores = []
    training_times = []
    for i in range(10):
        if i == 9:
            accuracy, f_score, training_time = train_basic_nn(all_data, patch_size, os.path.join(save_dir, str(i + 1)), True)
        else:
            accuracy, f_score, training_time = train_basic_nn(all_data, patch_size, os.path.join(save_dir, str(i + 1)))
        accuracies.append(accuracy)
        f_scores.append(f_score)
        training_times.append(training_time)
    return accuracies, f_scores, training_times


# normal cnn
def test_normal_cnn(patch_size, kernel_size, data_path, save_dir):
    # Read data
    all_data = pd.read_csv(data_path, header=None)

    accuracies = []
    f_scores = []
    training_times = []
    for i in range(10):
        if i == 9:
            accuracy, f_score, training_time = train_normal_cnn(all_data, patch_size, kernel_size, os.path.join(save_dir, str(i + 1)), True)
        else:
            accuracy, f_score, training_time = train_normal_cnn(all_data, patch_size, kernel_size, os.path.join(save_dir, str(i + 1)))
        accuracies.append(accuracy)
        f_scores.append(f_score)
        training_times.append(training_time)
    return accuracies, f_scores, training_times


# dilated cnn
def test_dilated_cnn(patch_size, kernel_size, data_path, save_dir):
    # Read data
    all_data = pd.read_csv(data_path, header=None)

    accuracies = []
    f_scores = []
    training_times = []
    for i in range(10):
        if i == 9:
            accuracy, f_score, training_time = train_dilated_cnn(all_data, patch_size, kernel_size, os.path.join(save_dir, str(i + 1)), True)
        else:
            accuracy, f_score, training_time = train_dilated_cnn(all_data, patch_size, kernel_size, os.path.join(save_dir, str(i + 1)))
        accuracies.append(accuracy)
        f_scores.append(f_score)
        training_times.append(training_time)
    return accuracies, f_scores, training_times


# basic nn
# accuracies, f_scores, training_times = test_basic_nn(patch_size=1,
#                                                      data_path='data/1by1_data_8bit.csv',
#                                                      save_dir='trained_model/basic_nn')

# basic nn with patch size 32
# accuracies, f_scores, training_times = test_basic_nn(patch_size=32,
#                                                      data_path='data/32by32_data_8bit.csv',
#                                                      save_dir='trained_model/basic_nn_32')

# normal cnn
# accuracies, f_scores, training_times = test_normal_cnn(patch_size=32, kernel_size=7,
#                                                        data_path='data/32by32_data_8bit.csv',
#                                                        save_dir='trained_model/normal_cnn_32_7')

# dilated cnn
accuracies, f_scores, training_times = test_dilated_cnn(patch_size=32, kernel_size=3,
                                                        data_path='data/32by32_data_8bit.csv',
                                                        save_dir='trained_model/dilated_cnn_32_3')


# Show the testing results
print("Accuracies for 10 times:")
print(accuracies)
print("Max accuracy:", round(np.max(accuracies), 3))
print("Average accuracy:", round(np.mean(accuracies), 3))
print("F-scores for 10 times:")
print(f_scores)
print("Max f-score:", round(np.max(f_scores), 3))
print("Average f-score:", round(np.mean(f_scores), 3))
print("Training time for 10 times:")
print(training_times)
print("Min training time:", round(np.min(training_times), 3))
print("Average training time:", round(np.mean(training_times), 3))