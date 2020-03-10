#!/usr/bin/env python3

"""
Train a custom model with a Resnet-50 as the base model.
Training occurs in steps.

The model and history are saved after the training is over

Model uses raw category in the generator

"""

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
import argparse
import logging
import logging.handlers

#MODEL_NAME = "ecom-image-model"
#LOG_FILE_NAME = MODEL_NAME + "_log.log"
FORMAT = '%(asctime)s [%(levelname)s] %(message)s'

TRAINING_DF = "image_classification_training_df.csv"
TESTING_DF = "image_classification_testing_df.csv"


def get_logger():
    # Initiate logging
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__file__)
    # Add a file logger
    file_handler = logging.handlers.RotatingFileHandler(filename=f"logs/{model_name}")
    file_handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(file_handler)
    # without this, logger doesn't work
    logger.setLevel(logging.INFO)

    return logger


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
# LOCAL_PATH = '/home/ilambda/debasish'

# ilambda2
# /home/ilambda/goods_viewer/Debasish/dataset
TRAIN_IMAGE_PATH = 'dataset/1_train_split/whole_resize'
TEST_IMAGE_PATH = 'dataset/1_eval_img_resize/'
MAPPED_LABELS_PATH = 'dataset/mappedcategory.txt'

BATCH_SIZE = 32


def create_or_load_custom_model(base_model, dropout=None):

    thresholds = [float(i / 100) for i in range(0, 101)]
    tp = keras.metrics.TruePositives(thresholds=thresholds, name="tp")
    tn = keras.metrics.TrueNegatives(thresholds=thresholds, name="tn")
    fp = keras.metrics.FalsePositives(thresholds=thresholds, name="fp")
    fn = keras.metrics.FalseNegatives(thresholds=thresholds, name="fn")
    precision = keras.metrics.Precision(thresholds=thresholds, name='precision')
    recall = keras.metrics.Recall(thresholds=thresholds, name='recall')
    auc = keras.metrics.AUC(name="auc", thresholds=thresholds)
    adam = keras.optimizers.Adam(learning_rate=lr)

    metrics = [tp, tn, fp, fn, precision, recall, auc]

    model = Sequential(name=model_name)
    model.add(base_model)
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(NUM_CATEGORIES, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=metrics)
    if model_path:
        try:
            logger.info(f"Loading previous model from path: {model_path}")
            previous_model = keras.models.load_model(model_path, compile=False)
        except FileNotFoundError:
            previous_model = None
            logger.info(f"Model not found at {model_path}")

    if previous_model:
        logger.info(f"Setting weights from previous model to new model")
        model.set_weights(previous_model.get_weights())

    logger.info(model.summary())
    return model


# create_custom_model(base_model)

def create_or_load_base_model(num_trained_layers=100):
    """

    :param num_trained_layers:
    :param pooling:
    :return:
    """
    if base_model_path:
        try:
            base_model = keras.models.load_model(base_model_path)
        except FileNotFoundError:
            logger.warning(f"Base model not found at location {base_model_path}")
            base_model = keras.applications.resnet50.ResNet50(weights='imagenet',
                                                              include_top=False,
                                                              input_shape=RGB_IMAGE_SIZE,
                                                              pooling='avg')
    else:
        base_model = keras.applications.resnet50.ResNet50(weights='imagenet',
                                                          include_top=False,
                                                          input_shape=RGB_IMAGE_SIZE,
                                                          pooling='avg')
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
        logger.error("[ERROR] Please specify a valid path!")
        return

    model = models.load_model(model_path)

    # Add a new metric
    # topkmetric = tf.keras.metrics.TopKCategoricalAccuracy(name="top5")
    if compile:
        model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics + new_metrics)

    model.summary()

    return model


def generate_file_name(name, type="--"):
    """

    :param name: Name of the file
    :param type: either history or model or base_model
    :return:
    """
    file_name = name + "_" + type + "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    if type == "history":
        file_name += ".json"
    else:
        file_name += ".h5"
    return file_name


def write_file_to_disk(file_name, contents):
    with open(file_name, 'w') as output:
        output.write(contents)
    logger.info(f"Data written to: {file_name}")


def train_model(model, training_gen, validation_gen):
    save_best_val_acc_model = ModelCheckpoint(filepath=f"{model.name}_best_val_acc",
                                              monitor="val_accuracy",
                                              mode="auto",
                                              save_best_only=True,
                                              verbose=1)
    save_best_val_loss_model = ModelCheckpoint(filepath=f"{model.name}_best_val_loss",
                                               monitor="val_loss",
                                               mode="auto",
                                               save_best_only=True,
                                               verbose=1)

    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

    csv_logger = keras.callbacks.CSVLogger(model.name + "training.csv")

    # configured to reduce LR based on val_loss
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  min_lr=0,
                                  patience=5,
                                  verbose=1,
                                  factor=.5)

    tensorboard = TensorBoard(log_dir=f"logs/{model_name}_{time.time()}")

    callbacks = [save_best_val_acc_model, save_best_val_loss_model, tensorboard, reduce_lr, csv_logger]

    # model = Sequential()
    history = model.fit(training_gen,
                        validation_data=validation_gen,
                        initial_epoch=initial_epoch,
                        epochs=epochs,
                        callbacks=callbacks,
                        shuffle=False)

    return history


def create_testing_data(test_df, class_mode='raw'):
    """

    :param test_df:
    :param class_mode:
    :return:
    """
    data_gen = image.ImageDataGenerator(rescale=1. / 255,
                                        preprocessing_function=keras.applications.resnet50.preprocess_input)
    test_gen = data_gen.flow_from_dataframe(dataframe=test_df,
                                            directory=TEST_IMAGE_PATH,
                                            x_col='filename',
                                            y_col='category',
                                            class_mode=class_mode,
                                            classes=[i for i in range(NUM_CATEGORIES)],
                                            target_size=IMAGE_SIZE,
                                            shuffle=False)
    return test_gen


def create_training_data(train_df, validation_split=0.3, class_mode='raw', shuffle=False):
    """

    :param train_df:
    :param validation_split:
    :param class_mode:
    :param shuffle:
    :return:
    """
    # Initialize a Image generator
    data_gen = image.ImageDataGenerator(rescale=1. / 255, validation_split=0.3,
                                        preprocessing_function=keras.applications.resnet50.preprocess_input)

    # Create the image generators to be fed into the model
    training_gen = data_gen.flow_from_dataframe(dataframe=train_df,
                                                directory=TRAIN_IMAGE_PATH,
                                                x_col='filename',
                                                y_col='category',
                                                classes=[i for i in range(NUM_CATEGORIES)],
                                                class_mode=class_mode,
                                                target_size=IMAGE_SIZE,
                                                subset='training',
                                                shuffle=shuffle)
    validation_gen = data_gen.flow_from_dataframe(dataframe=train_df,
                                                  directory=TRAIN_IMAGE_PATH,
                                                  x_col='filename',
                                                  y_col='category',
                                                  classes=[i for i in range(NUM_CATEGORIES)],
                                                  class_mode=class_mode,
                                                  target_size=IMAGE_SIZE,
                                                  subset='validation',
                                                  shuffle=False)
    return training_gen, validation_gen


def plot_results(history):
    if not isinstance(history, dict):
        logger.error(f"[ERROR] The input object must be of type - dict()")
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

    logger.info(results)


def lr_schedule(epoch):
    """
  Returns a custom learning rate that decreases as epochs progress.
  """
    learning_rate = 0.2
    if epoch > 10:
        learning_rate = 0.02
    if epoch > 20:
        learning_rate = 0.01
    if epoch > 50:
        learning_rate = 0.005

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


def load_data_frame(filename, directory="."):
    abs_path = os.path.abspath(directory)
    df_path = os.path.join(abs_path, filename)
    if not os.path.exists(df_path):
        logger.error(f"Dataframe path {df_path} doesn't exist!")
        exit(1)
    else:
        df = pd.read_csv(df_path, index_col='index')

    return df


def main():
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
        {"filename": [f[0] for f in train_mapped_categories], "category": [int(f[1]) for f in train_mapped_categories]})
    test_df = pd.DataFrame(
        {"filename": [f[0] for f in test_mapped_categories], "category": [int(f[1]) for f in test_mapped_categories]})
    # test_df.head()

    # Write the data-frames and the dictionary to disk, so that it can be reused

    train_df.to_csv(TRAINING_DF, index_label='index')
    test_df.to_csv(TRAINING_DF, index_label='index')
    write_file_to_disk("image_classification_categories.json", mapped_categories)

    logger.info(f"Shape of train dataframe: {train_df.shape}")
    logger.info(f"Shape of test dataframe: {test_df.shape}")

    training_gen, validation_gen = create_training_data(train_df,
                                                        validation_split=0.3,
                                                        class_mode='raw')

    base_model = create_or_load_base_model(num_trained_layers=0)

    # Add a new metric

    # sparse metrics will only work if the loss function is sparse
    new_metrics = ['accuracy', keras.metrics.top_k_categorical_accuracy]

    adam = keras.optimizers.Adam(learning_rate=0.0001)

    # In case of sparse, make sure that the generator class_mode is also sparse
    # for categorical loss, generator class_mode is categorical
    model = create_or_load_custom_model(base_model)

    history = train_model(model=model,
                          training_gen=training_gen,
                          validation_gen=validation_gen,
                          total_epochs=epochs)

    write_file_to_disk(generate_file_name(model_name, type="history"))
    model.save(generate_file_name(model_name, type="model"))
    base_model.save(generate_file_name(model_name, type="basemodel"))

    logger.info("Training complete!!")
    logger.info(history.history)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, type=str, help="Name of the model")
    parser.add_argument("--epochs", required=False, default=10, type=int, help="Number of epochs to train the model")
    parser.add_argument("--init-epoch", required=False, default=0, type=int,
                        help="Initial epochs at which the training to begin")
    parser.add_argument("--base-trainable", required=True, default=0, type=int,
                        help="0: Non trainable, -1: All layers trainable, Positive Number: No. of layers trainable. Default: 0")
    parser.add_argument("--dataset-path", required=False, default="./data",
                        help="The full path where the training and testing dataset is located")
    parser.add_argument("--model-path", default=None, required=False,
                        help="The path of the model if the training needs to be performed on an existing model")
    parser.add_argument("--base_model_path", default=None, required=False, type=str,
                        help="Path of the saved base model to load")
    parser.add_argument("--lr", default=0.001, required=False, type=float,
                        help="Learning rate for the ADAM optimizer. Default is 0.001")

    args = parser.parse_args()

    model_name = args.model_name
    epochs = args.epochs
    initial_epoch = args.init_epoch
    dataset_path = args.dataset_path
    base_trainable = args.base_trainable
    lr = args.lr
    base_model_path = args.base_model_path
    model_path = args.model_path

    logger = get_logger()

    logger.info(f"The following arguments are passed to the model\n{args}")



    main()
