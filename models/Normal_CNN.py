import os
import joblib
import tensorflow as tf
import time

from tensorflow import keras
from utils import pre_process_cnn_data
from utils import plot_learning_curves
from sklearn.metrics import f1_score


# Define basic CNN model
def define_basic_model(patch_size, kernel_size):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=128, padding='same',
                                  kernel_size=kernel_size,
                                  input_shape=(patch_size, patch_size, 5)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))

    model.add(keras.layers.Conv2D(filters=256, padding='same',
                                  kernel_size=kernel_size))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))

    return model


def train_normal_cnn(all_data, patch_size, kernel_size, save_dir, is_plot=False):

    # Load data
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test), scaler, pca = pre_process_cnn_data(patch_size=patch_size, all_data=all_data)

    # Define normal cnn model
    model = define_basic_model(patch_size, kernel_size)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])
    model.summary()

    start = time.clock()
    # Train model
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    output_model_file = os.path.join(save_dir, "model.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True),
        keras.callbacks.EarlyStopping(patience=30, min_delta=1e-4),
    ]
    history = model.fit(x_train, y_train, epochs=300, validation_data=(x_valid, y_valid), callbacks=callbacks, shuffle=True, batch_size=8)
    end = time.clock()
    training_time = end - start

    # Plot training results
    if is_plot:
        plot_learning_curves(history)

    # Test model
    del model
    model = keras.models.load_model(os.path.join(save_dir, 'model.h5'))
    print('Testing results:')
    accuracy = model.evaluate(x_test, y_test)[1]
    y_test_pred = tf.argmax(model.predict(x_test), axis=1)
    f_score = f1_score(y_test, y_test_pred, average='weighted')

    # Save scaler and pca
    joblib.dump(scaler, os.path.join(save_dir, 'scaler'))
    joblib.dump(pca, os.path.join(save_dir, 'pca'))

    return accuracy, f_score, training_time
