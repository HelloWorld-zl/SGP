import numpy as np
import cv2 as cv
import os
import tensorflow as tf
import tensorflow.keras as keras
import joblib
import gc
import time
import matplotlib.image as mpl_img


# Path for test image
data_path = 'data\\test_image'
# Area for test image
row_min = 3000
row_max = 4000
column_min = 2500
column_max = 4000
row_num = row_max - row_min
column_num = column_max - column_min


# Select area for 23 multi-spectral images
images = np.zeros((row_num, column_num, 0))
for root, dirs, image_names in os.walk(data_path):
    for image_name in image_names:
        img = cv.imread(os.path.join(data_path, image_name), -1)
        img = np.expand_dims(img[row_min: row_max, column_min: column_max], axis=2)
        images = np.concatenate((images, img), 2)
images = tf.constant(images, dtype=tf.uint8)

# show original image
cv.namedWindow("Original image", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
cv.imshow("Original image", images[:, :, 1].numpy())
cv.waitKey(0)

# Load model and scaler
path = 'trained_model/basic_nn/10'  # the best one is 10th of all 10 models
model = keras.models.load_model(os.path.join(path, 'model.h5'))
scaler = joblib.load(os.path.join(path, 'scaler'))
pca = joblib.load(os.path.join(path, 'pca'))
print(model)

# Classification
start = time.clock()
rgb_image = np.zeros((0, 3))
for x_index in range(0, row_num):
    if x_index % 100 == 0:
        print('Progress:', x_index, '/', row_num)
    features_of_one_row = tf.reshape(images[x_index, :, :], [column_num, 23])
    features_of_one_row_scaled = scaler.transform(features_of_one_row)
    features_of_one_row_pca = pca.transform(features_of_one_row_scaled)
    labels_of_one_row = model.predict(features_of_one_row_pca)
    rgb_image = np.concatenate((rgb_image, labels_of_one_row), axis=0)
    del features_of_one_row
    del labels_of_one_row
    del features_of_one_row_scaled
    gc.collect()
end = time.clock()

rgb_image = rgb_image.reshape((row_num, column_num, 3))

cv.namedWindow("Enhanced image", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
cv.imshow('Enhanced image', rgb_image)
cv.waitKey(0)

print("Time used:", end - start)
mpl_img.imsave("out.png", rgb_image[:, :, ::-1])
