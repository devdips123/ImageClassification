#!/usr/bin/env python3

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import time
import pathlib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

keras = tf.keras
models = tf.keras.models
image = tf.keras.preprocessing.image
models = keras.models

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
RGB_IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
NUM_CATEGORIES = 68

# ilambda2
LOCAL_PATH = '/home/ilambda/goods_viewer/Debasish'
# ilambda3
#LOCAL_PATH = '/home/ilambda/debasish'

# ilambda2
#/home/ilambda/goods_viewer/Debasish/dataset
TRAIN_IMAGE_PATH = LOCAL_PATH + '/dataset/1_train_split/whole_resize'
TEST_IMAGE_PATH = LOCAL_PATH + '/dataset/1_eval_img_resize/'
MAPPED_LABELS_PATH = LOCAL_PATH + '/dataset/mappedcategory.txt'

BATCH_SIZE = 32

print(f"Tensorflow version: {tf.__version__}")


def create_custom_model(base_model, metrics=['accuracy'], activation='softmax',
                        optimizer='adam', loss='sparse_categorical_crossentropy',
                        name='deb_resnet', dropout=0.1):
    model = Sequential(name=name)
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(dropout))
    model.add(Dense(NUM_CATEGORIES, activation=activation))
    # adam = Adam(learning_rate=lr)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    print(model.summary())
    return model


# create_custom_model(base_model)

def create_resnet_model(input_shape=RGB_IMAGE_SIZE, include_top=False, num_trained_layers=100):
    """

    :param input_shape:
    :param include_top:
    :param num_trained_layers: how many layers to un-freeze.
        -1 means unfreeze all
        0 means freeze all layers
    :return: resnet_model
    """
    base_model = keras.applications.resnet50.ResNet50(weights='imagenet',
                                                      include_top=include_top,
                                                      input_shape=input_shape)
    if num_trained_layers == 0:
        base_model.trainable = False
        return base_model

    base_model.trainable = True
    if num_trained_layers > 0:
        for i, layer in enumerate(base_model.layers):
            if i < num_trained_layers:
                layer.trainable = False

    return base_model


def load_saved_model(model_path, compile=False, new_metrics=[]):
    if not os.path.exists(model_path):
        print("[ERROR] Please specify a valid path!")
        return

    model = models.load_model(model_path)

    # Add a new metric
    # topkmetric = tf.keras.metrics.TopKCategoricalAccuracy(name="top5")
    if compile:
        model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics + new_metrics)

    model.summary()

    return model


def generate_file_name(name, type="--"):
    return name + "_" + type + "_" + time.strftime("%Y_%m_%d_%M_%S")


def train_model(model, training_gen, validation_gen, initial_epoch=0, total_epochs=5, save_history=True, callbacks=[]):
    # model = Sequential()
    history = model.fit_generator(training_gen,
                                  validation_data=validation_gen,
                                  initial_epoch=initial_epoch,
                                  epochs=total_epochs,
                                  callbacks=callbacks,
                                  shuffle=False)
    if save_history:
        name = model.name
        file_name = generate_file_name(model.name, type="history") + ".json"
        output = open(file_name, 'w')
        output.write(json.dumps(str(history.history)))
        output.close()
        print(f"Model history written to file: {file_name}")

    return history


def create_testing_data(test_df, class_mode='sparse'):
    data_gen = image.ImageDataGenerator(rescale=1. / 255)
    test_gen = data_gen.flow_from_dataframe(dataframe=test_df,
                                            directory=TEST_IMAGE_PATH,
                                            x_col='filename',
                                            y_col='category',
                                            class_mode=class_mode,
                                            classes=[str(i) for i in range(NUM_CATEGORIES)],
                                            target_size=IMAGE_SIZE)
    return test_gen


def create_training_data(train_df, validation_split=0.3, class_mode='sparse'):
    # Initialize a Image generator
    data_gen = image.ImageDataGenerator(rescale=1. / 255, validation_split=0.3)

    # Create the image generators to be fed into the model
    training_gen = data_gen.flow_from_dataframe(dataframe=train_df,
                                                directory=TRAIN_IMAGE_PATH,
                                                x_col='filename',
                                                y_col='category',
                                                classes=[str(i) for i in range(NUM_CATEGORIES)],
                                                class_mode=class_mode,
                                                target_size=IMAGE_SIZE,
                                                subset='training')
    validation_gen = data_gen.flow_from_dataframe(dataframe=train_df,
                                                  directory=TRAIN_IMAGE_PATH,
                                                  x_col='filename',
                                                  y_col='category',
                                                  classes=[str(i) for i in range(NUM_CATEGORIES)],
                                                  class_mode=class_mode,
                                                  target_size=IMAGE_SIZE,
                                                  subset='validation')
    return training_gen, validation_gen


def plot_results(history):
    if not isinstance(history, dict):
        print(f"[ERROR] The input object must be of type - dict()")
        return

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def evaluate_model(model, validation_gen):
    # evaluate the model
    results = model.evaluate(x=validation_gen)

    print(results)


# Prepare data

mapped_categories = np.loadtxt(MAPPED_LABELS_PATH, dtype='int', delimiter=',')
mapped_categories = dict(mapped_categories)

# Name of the files in the test directory
test_files = [f.name for f in pathlib.Path(TEST_IMAGE_PATH).glob('*.jpg')]
train_files = [f.name for f in pathlib.Path(TRAIN_IMAGE_PATH).glob('*.jpg')]
# Map the test file names to the categories
train_mapped_categories = [(str(f), str(mapped_categories.get(int(f.split('.')[0])))) for f in train_files]
test_mapped_categories = [(str(f), str(mapped_categories.get(int(f.split('.')[0])))) for f in test_files]
# Create a dataframe
train_df = pd.DataFrame(
    {"filename": [f[0] for f in train_mapped_categories], "category": [f[1] for f in train_mapped_categories]})
test_df = pd.DataFrame(
    {"filename": [f[0] for f in test_mapped_categories], "category": [f[1] for f in test_mapped_categories]})
# test_df.head()

print(f"Shape of train dataframe: {train_df.shape}")
print(f"Shape of test dataframe: {test_df.shape}")


training_gen, validation_gen = create_training_data(train_df, validation_split=0.3)
"""

"""
base_model = create_resnet_model()

# Add a new metric
# topkmetric = tf.keras.metrics.TopKCategoricalAccuracy(name="top5")
new_metrics = ['accuracy', keras.metrics.sparse_categorical_accuracy, keras.metrics.sparse_top_k_categorical_accuracy,
               keras.metrics.top_k_categorical_accuracy]

adam = keras.optimizers.Adam(learning_rate=0.0001)
#loss=keras.losses.categorical_crossentropy
model = create_custom_model(base_model=base_model, metrics=new_metrics, optimizer=adam)

save_model = ModelCheckpoint(filepath="resnet_best_val_acc",
                             monitor="val_accuracy",
                             mode="auto",
                             save_best_only=True,
                             verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                              min_lr=0,
                              patience=5,
                              verbose=1,
                              factor=.5)

tensorboard = TensorBoard(log_dir=f"logs/resnet_{time.time()}")

callbacks = [save_model, tensorboard]

EPOCHS = 5

history = train_model(model=model,
                      training_gen=training_gen,
                      validation_gen=validation_gen,
                      total_epochs=EPOCHS,
                      callbacks=callbacks)

model.save(generate_file_name(model.name, type="model") + ".h5")

#plot_results(history)

