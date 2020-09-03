import numpy as np
import pandas as pd
import os
import cv2 as cv
import math
import re
import csv


patch_radius = 16

csv_path = './data/training_file_8_bit.txt'
csv_data = pd.read_csv(csv_path)

# Replace class_name with integer value: class_m -> 0, class_o -> 1, class_u -> 2
csv_data['class_name'] = csv_data['class_name'].map(lambda x: re.sub('class_m', '0', x))
csv_data['class_name'] = csv_data['class_name'].map(lambda x: re.sub('class_o', '1', x))
csv_data['class_name'] = csv_data['class_name'].map(lambda x: re.sub('class_u', '2', x))

# data_path = 'G:\\Syriac_Y\\Syriac_Flattened-TIFFs_Full_16bit\\'
data_path = 'G:\\Syriac_Y\\Syriac_TIFFs_8bit\\'

data_with_neighbours = np.zeros([0, 0])
print(data_with_neighbours, type(data_with_neighbours))

all_samples = np.zeros(([0, 3 + int(math.pow(patch_radius * 2, 2)) * 23]))

for folder_name, samples in csv_data.groupby('folio_name'):

    # If the file exists in the data file, that is, there is labeled data in this folio
    print('Folder "', folder_name, '" exists:', os.path.exists(data_path + folder_name))
    if os.path.exists(data_path + folder_name):

        # Find locations of samples in this folio
        position_x = samples['x_loc'].values
        position_y = samples['y_loc'].values
        labels = samples['class_name'].values

        # Create a set of samples
        samples_in_one_folio = np.hstack((position_x.reshape(-1, 1), position_y.reshape(-1, 1), labels.reshape(-1, 1)))

        # Traverse all multi-spectral images
        for root, dirs, image_names in os.walk(data_path + folder_name):
            for image_name in image_names:
                img = cv.imread(root + '\\' + image_name, -1)

                features_for_one_image = np.zeros([0, int(math.pow(patch_radius * 2, 2))])
                # For each multi-spectral image, extract neighbours' values of all samples
                # Shape: (num_samples_for_one_folio, kernel_size * kernel_size)
                for i in range(len(position_x)):
                    features_of_neighbours = img[position_y[i] - patch_radius: position_y[i] + patch_radius,
                                             position_x[i] - patch_radius: position_x[i] + patch_radius].reshape(-1)
                    features_for_one_image = np.vstack((features_for_one_image, features_of_neighbours))

                # Concatenate values for 23 multi-spectral images
                # Shape: (num_samples_for_one_folio, kernel_size * kernel_size * 23)
                samples_in_one_folio = np.hstack((samples_in_one_folio, features_for_one_image))

        print('Data in this folio:', samples_in_one_folio, samples_in_one_folio.shape)

        # Concatenate samples from different folios
        # Shape: (num_samples, 3 + kernel_size * kernel_size * 23)
        all_samples = np.vstack((all_samples, samples_in_one_folio))
        print('All data:', all_samples, all_samples.shape)

data_dir = './data'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

data_file_name = '32by32_data_8bit.csv'

data_path = os.path.join(data_dir, data_file_name)
with open(data_path, "w+", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_samples)
    f.close()