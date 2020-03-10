#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import logging.handlers
import os
import pathlib
import re
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
RGB_IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
BATCH_SIZE = 32
# BASE_MODEL_NAME = 'trained_resnet_model.h5'

keras = tf.keras

TRAIN_IMAGE_PATH = '/home/ilambda/goods_viewer/Debasish/dataset/1_train_split/whole_resize'
TEST_IMAGE_PATH = '/home/ilambda/goods_viewer/Debasish/dataset/1_eval_img_resize/'

FORMAT = '%(asctime)s [%(levelname)s] %(message)s'


# ### Data analysis tasks
# 
# - Create a dataframe of image_name to labels input file (labels_map_df)
# - Import the features file as a list (features_list)
# - Filter the dataframe columns to retain all the columns matching in features_file and discard the rest. Copy the resulting dataframe to a new dataframe (updated_labels_df)
#     - Make sure to add the columns ['#Attr 266', 'Name', 'Price']
#     
# Create the following plots
# 
# -


def configure_logger():
    # Initiate logging
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__file__)
    # Add a file logger
    file_handler = logging.handlers.RotatingFileHandler(filename=LOG_FILE_NAME)
    file_handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(file_handler)
    # without this, logger doesn't work
    logger.setLevel(logging.INFO)

    return logger


def write_file_to_disk(file_name, contents):
    with open(file_name, 'w') as output:
        output.write(contents)
    logger.info(f"Model history written to file: {file_name}")


def generate_file_name(name, type="--"):
    return name + "_" + type + "_" + time.strftime("%Y_%m_%d_%H_%M_%S")


def load_data_frame(filename, directory="."):
    abs_path = os.path.abspath(directory)
    df_path = os.path.join(abs_path, filename)
    if not os.path.exists(df_path):
        logger.error(f"Dataframe path {df_path} doesn't exist!")
        exit(1)
    else:
        df = pd.read_csv(df_path, index_col='index')

    return df


def write_list_to_disk(filename, plist, directory="."):
    if not os.path.exists(filename):
        abs_path = os.path.abspath(directory)
        filename = os.path.join(abs_path, filename)
    if not os.path.exists(filename):
        logger.error(f"File path not found: {filename}\n List not written")
        return
    with open(filename, 'w') as filehandle:
        for listitem in plist:
            filehandle.write('%s\n' % listitem)
    logger.info(f"List written to {filename}")


def read_list_from_disk(filename, directory="."):
    plist = []
    if not os.path.exists(filename):
        abs_path = os.path.abspath(directory)
        filename = os.path.join(abs_path, filename)
    if not os.path.exists(filename):
        logger.error(f"File path not found: {filename}\n List could not be retrieved")
        # exit(-1)
    with open(filename, 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            l = line[:-1]
            # add item to the list
            plist.append(l)
    logger.info(f"List read from {filename}")

    return plist


def write_list_to_disk(filename, plist, directory="."):
    # if not os.path.exists(filename):
    # abs_path = os.path.abspath(directory)
    # filename = os.path.join(abs_path, filename)
    with open(filename, 'w') as filehandle:
        for listitem in plist:
            filehandle.write('%s\n' % listitem)
    logger.info(f"List written to {filename}")


def find_feature_columns(df):
    all_cols = set(df.columns)
    feature_cols = all_cols.difference(set(['filename', 'Name', 'Price']))
    return sorted(list(feature_cols))


def filter_features(df, threshold=0.01):
    """
    Filter features based on sparsity
    :param df:
    :param threshold:
    :return:
    """
    feature_cols = find_feature_columns(df)
    # features_df is a Series if we have only 1 measure np.mean else df if [np.mean, len]
    features_df = df[feature_cols].apply(np.mean, axis=0).T
    filtered_features = features_df[features_df > threshold].index
    logger.info(f"The number of features with mean 1s > {threshold}: {len(filtered_features)}")
    return filtered_features


# logger = configure_logger()


def create_train_test_dfs():
    """
    This function creates the training and testing data from the scratch by
    reading the selected_features.txt which contains the list of image names with its features
    and selected_feature.txt which contains the name of 229 most widely-used features.
    This function, reads the above files and then filters the image_names from the file by comparing
    with the images in the training and testing directories.
    It performs pre-proessing, removes duplicates, removes rows with nans, etc.
    Returns two separate data-frames for training and testing and also the list of features or labels
    :return: training_df, testing_df, labels
    """

    labels = pd.read_csv('dataset/jc_input.txt')
    labels.head()
    logger.info(f"Shape of the dataset: {labels.shape}")

    features = np.loadtxt('dataset/selected_feature.txt', dtype=str, delimiter='\n')
    logger.info(f"Number of features in features.txt: {len(features)}")

    # ### Change the column names in dataset. Repelace pipe with space.
    # This is done to match the column names specified in  the selected_features.txt
    columns_with_pipe = list(filter(lambda x: re.match('.*\|.*', x), labels.columns))
    len(columns_with_pipe)
    columns_replaced_pipe = set(map(lambda x: x.replace('|', ' '), labels.columns))

    len(columns_replaced_pipe.intersection(set(features)))

    # ### List of columns containing the labels
    # Sorted alphabetically
    label_columns = sorted(list(columns_replaced_pipe.intersection(set(features))))
    all_columns = ["#Attr 266", "Name", "Price"] + label_columns

    updated_labels_df = labels.copy()

    # ### Stores the mapping of renamed_cols to original_cols

    renamed_to_orig_cols_dict = dict([(col, col.replace('|', ' ')) for col in updated_labels_df.columns])

    updated_labels_df.rename(renamed_to_orig_cols_dict, axis=1, inplace=True)

    updated_labels_df = updated_labels_df[all_columns]

    updated_labels_df.shape

    # ### Update the column name "#Attr 266" to filename
    updated_labels_df.rename({'#Attr 266': "filename"}, axis=1, inplace=True)
    updated_labels_df["filename"] = updated_labels_df["filename"].apply(lambda x: str(x) + ".jpg")

    training_list = [f.name for f in pathlib.Path(TRAIN_IMAGE_PATH).glob('*.jpg')]
    testing_list = [f.name for f in pathlib.Path(TEST_IMAGE_PATH).glob('*.jpg')]

    # In[29]:

    training_fname = list(set(updated_labels_df.filename.values).intersection(set(training_list)))
    testing_fname = list(set(updated_labels_df.filename.values).intersection(set(testing_list)))
    logger.info(f"Number of training files with attributes: {len(training_fname)}")
    logger.info(f"Number of testing files with attributes: {len(testing_fname)}")

    # ### Remove duplicates

    # In[30]:

    # Identify duplicate files
    updated_labels_df.filename.value_counts()

    # In[31]:

    # 4 files with name '&  ress.jpg' are duplicated
    to_remove_index = updated_labels_df[(updated_labels_df.filename == '&  ress.jpg')].index

    # In[32]:

    updated_labels_df.drop(to_remove_index, inplace=True)

    # ### Remove nans
    rows_with_nan = updated_labels_df[updated_labels_df[label_columns].isna().sum(axis=1) > 0].index
    logger.info(f"Number of rows with NaN: {len(rows_with_nan)}")
    updated_labels_df.drop(rows_with_nan, inplace=True)
    updated_labels_df.shape

    # Check for further nans
    sum(updated_labels_df[label_columns].isna().sum() > 0)
    # ### Update the Nan name attribute with blank_string
    updated_labels_df.Name.fillna("", inplace=True)

    # ### Train test split
    # - update index to dataframe to file name
    # - perform a set intersection of df index and training_fname from the training images directory
    # - perform a set intersection of df index and testing_fname from the testing images directory
    # - create separate dfs - training_df and testing_df

    updated_labels_df.set_index(updated_labels_df.filename, inplace=True)

    train_index = set(updated_labels_df.index).intersection(set(training_fname))
    test_index = set(updated_labels_df.index).intersection(set(testing_fname))

    # ### Training and testing df
    training_df = updated_labels_df.loc[list(train_index)]
    testing_df = updated_labels_df.loc[list(test_index)]

    logger.info(f"Training df: {training_df.shape}")
    logger.info(f"Testing df: {testing_df.shape}")

    return training_df, testing_df, label_columns


def create_new_model(base_model_name=None, train_base=False,
                     dropout=None, lr=0.001, class_weights=None):
    logger.info(f"Creating new model with name: {model_name}")
    if base_model_name:
        try:
            logger.info(f"Loading base model: {base_model_name}")
            base_model = keras.models.load_model(base_model_name)
        except Exception:
            logger.warning(
                f"Error occurred while loading the base model from {base_model_name}\nCreating new base model with imagenet weights")
            base_model = keras.applications.resnet50.ResNet50(weights='imagenet',
                                                              include_top=False,
                                                              input_shape=RGB_IMAGE_SIZE,
                                                              pooling='avg')
    else:
        base_model = keras.applications.resnet50.ResNet50(weights='imagenet',
                                                          include_top=False,
                                                          input_shape=RGB_IMAGE_SIZE,
                                                          pooling='avg')
    base_model.trainable = train_base
    logger.info(f"Base Model trainable = {base_model.trainable}")
    model = keras.models.Sequential(name=model_name)
    model.add(base_model)

    model.add(keras.layers.Dense(1024, activation='relu'))
    if dropout:
        model.add(keras.layers.Dropout(rate=dropout))
    model.add(keras.layers.Dense(num_categories, activation="sigmoid"))

    thresholds = [float(i / 100) for i in range(0, 101)]
    tp = keras.metrics.TruePositives(thresholds=thresholds, name="tp")
    tn = keras.metrics.TrueNegatives(thresholds=thresholds, name="tn")
    fp = keras.metrics.FalsePositives(thresholds=thresholds, name="fp")
    fn = keras.metrics.FalseNegatives(thresholds=thresholds, name="fn")
    precision = keras.metrics.Precision(thresholds=thresholds, name='precision')
    recall = keras.metrics.Recall(thresholds=thresholds, name='recall')
    auc = keras.metrics.AUC(name="auc", thresholds=thresholds)
    adam = keras.optimizers.Adam(learning_rate=lr)
    rms = keras.optimizers.RMSprop(learning_rate=lr)
    sgd = keras.optimizers.SGD(learning_rate=lr)

    # Loss function with class-weights
    if isinstance(class_weights, np.ndarray):
        loss_function = get_weighted_loss(class_weights)
        logger.info(f"Using custom loss function with class weights")
    else:
        loss_function = 'binary_crossentropy'

    logger.info(f"The loss function used is : {loss_function}")
    model.compile(optimizer=adam,
                  loss=loss_function,
                  metrics=['accuracy', tp, tn, fp, fn, precision, recall, auc])
    # model.metrics.
    logger.info(f"Model Summary\n{model.summary()}")

    return model


def load_existing_model(model_path, class_weights=None):
    if not os.path.exists(model_path):
        abs_path = os.path.abspath(".")
        model_path = os.path.join(abs_path, model_path)
        if not os.path.exists(model_path):
            logger.error(f"Path {model_path} doesn't exist")
            exit(1)

    logger.info(f"Loading existing model from disk at {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    weights = model.get_weights()
    new_model = create_new_model(base_model_name=base_model_path, train_base=base_trainable,
                                 class_weights=class_weights)
    logger.info(f"Setting the weights of the loaded model to the new model")
    new_model.set_weights(weights)
    model = new_model
    """
    #adam = keras.optimizers.Adam(learning_rate=0.0001)
    try:
        model.compile(optimizer=adam, loss=model.loss, metrics=model.metrics)
    except AttributeError:
        logger.warning(f"The loaded model failed to compile. Creating a new model and setting the weights.")
        weights = model.get_weights()
        new_model = create_new_model(base_trainable=True)
        new_model.set_weights(weights)
        model = new_model
    """
    logger.info(f"Model Summary\n{model.summary()}")

    return model


def calculate_class_weights(df, features, classes_per_feature=[0., 1.], weight_type=0.5):
    """
    Calculates the class weights based on the distribution of the classes in the given samples in the dataframe.
    Multiplies the positive class weight with a factor as specified in weight_type. For 'balanced', weight_type=1
    Class weight is calculated as num_samples/(num_classes * np.bincount(feature))
    :param df: Dataframe containing the training samples
    :param features: A list containing the set of features in the dataframe.
    :param classes_per_feature: As the name suggests its the number of classes per feature
    :param weight_type: a factor to be multiplied to the positive class
    """
    num_features = len(features)
    num_classes = len(classes_per_feature)
    class_weights = np.zeros(shape=(num_features, num_classes))
    for i, feature in enumerate(features):
        feature_i = np.array(df[feature].values, dtype=int)
        weights_i = feature_i.shape[0] / (num_classes * np.bincount(feature_i))
        weights_i[1] = weights_i[1] * weight_type
        class_weights[i] = weights_i

    return class_weights


"""
def calculate_class_weights(df, features, classes=[0., 1.]):
 
    Computes the class weights for each label in the multi-label classification
    :param df: data-frame containing the data
    :param features: The list of features or the labels
    :param classes: target classes for each label. It is [0,1]
    :return: A ndarray of class-weights


    from sklearn.utils.class_weight import compute_class_weight
    num_features = len(features)
    num_classes = len(classes)
    class_weights = np.empty((num_features, num_classes))
    for i in range(num_features):
        class_weights[i] = compute_class_weight('balanced', classes, df[features[i]].values)

    return class_weights
"""


def get_weighted_loss(weights):
    """
    Loss function with class weights
    :param weights: a dictionary containing class weights ( for class 0 and class 1) for each label
    :return: loss
    """

    def weighted_loss(y_true, y_pred):
        return K.mean(
            (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** y_true) * K.binary_crossentropy(y_true, y_pred),
            axis=-1)

    return weighted_loss


def main(use_class_weights=False):
    # Read the training and testing data-frames from the disk
    global num_categories
    logger.info(f"Preparing data for the model")
    training_df_path = "attributes_training_df.csv"
    testing_df_path = "attributes_testing_df.csv"
    training_df = load_data_frame(training_df_path)
    testing_df = load_data_frame(testing_df_path)
    logger.info(f"Training df: {training_df.shape}")
    logger.info(f"Testing df: {testing_df.shape}")

    filtered_label_columns = filter_features(training_df)
    label_columns = filtered_label_columns
    logger.info(f"The list of filtered features:\n{label_columns}")

    # Write the list to file
    features_fname = "filtered_features.txt"
    logger.info(f"Writing the features list to the disk: {features_fname}")
    write_list_to_disk(features_fname, label_columns)
    num_categories = len(label_columns)
    logger.info(f"Number of features: {num_categories}")

    if use_class_weights:
        # Class weights
        class_weights = calculate_class_weights(training_df, label_columns)
        cw_filename = f"class_weights_{num_categories}"
        logger.info(f"Saving class weights to {cw_filename}")
        np.save(cw_filename, class_weights)
    else:
        class_weights = None

    # ### Create generators

    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                           validation_split=0.3,
                                                           preprocessing_function=keras.applications.resnet50.preprocess_input)
    testdatagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                               preprocessing_function=keras.applications.resnet50.preprocess_input)

    # use class_mode as 'multi_output' or 'other'

    # for other - each label would be 229 dimensions
    training_gen = datagen.flow_from_dataframe(training_df,
                                               directory=TRAIN_IMAGE_PATH,
                                               x_col='filename',
                                               y_col=label_columns,
                                               class_mode='other',
                                               target_size=IMAGE_SIZE,
                                               subset='training',
                                               shuffle=False)

    validation_gen = datagen.flow_from_dataframe(training_df,
                                                 directory=TRAIN_IMAGE_PATH,
                                                 x_col='filename',
                                                 y_col=label_columns,
                                                 class_mode='other',
                                                 target_size=IMAGE_SIZE,
                                                 subset='validation',
                                                 shuffle=False)

    test_gen = testdatagen.flow_from_dataframe(testing_df,
                                               directory=TEST_IMAGE_PATH,
                                               x_col='filename',
                                               y_col=label_columns,
                                               class_mode='other',
                                               target_size=IMAGE_SIZE,
                                               shuffle=False)
    logger.info(f"The model path is : {model_path}")
    if model_path:
        model = load_existing_model(model_path, class_weights=class_weights)
    else:
        # model = create_new_model(train_base=base_trainable)
        model = create_new_model(base_model_name=base_model_path, train_base=base_trainable,
                                 class_weights=class_weights, dropout=0.2)

    # epochs = 10
    logger.info(f"Total number of epochs: {epochs}")

    tensorboard = keras.callbacks.TensorBoard(
        log_dir=f'./logs/{model_name}',
        histogram_freq=1,
        write_images=True
    )

    save_best_val_loss_model = keras.callbacks.ModelCheckpoint(filepath=f"{model_name}_best_val_loss",
                                                               monitor="val_loss",
                                                               mode="auto",
                                                               save_best_only=True,
                                                               verbose=1)

    save_best_val_auc_model = keras.callbacks.ModelCheckpoint(filepath=f"{model_name}_best_val_auc",
                                                              monitor="val_auc",
                                                              mode="max",
                                                              save_best_only=True,
                                                              verbose=1)

    history = model.fit(training_gen, validation_data=validation_gen, verbose=1, epochs=epochs,
                        initial_epoch=initial_epochs, shuffle=False,
                        callbacks=[tensorboard, save_best_val_loss_model, save_best_val_auc_model])
    name = generate_file_name(model_name, type="model") + ".h5"
    logger.info(f"Saving the model as : {name}")
    model.save(name)

    # Save history to disk
    file_contents = json.dumps(str(history.history))
    write_file_to_disk(generate_file_name(model_name, type="history") + ".json", file_contents)
    logger.info(history.history)
    logger.info(f"Training complete")


if __name__ == '__main__':
    # Global variables
    num_categories = None
    model_name = None
    epochs = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",
                        help="name of the model: r- retraining, b: base trainable, in the form of filtered_attributes_<i>_b_r",
                        required=True)
    parser.add_argument("--epochs", default=10, type=int, required=False)
    parser.add_argument("--init-epochs", default=0, type=int, required=True,
                        help="Initial epoch at which model should start training. Used in case of re-training")
    parser.add_argument("--model-path", default=None, required=False,
                        help="The path of the previous trained model to load")
    parser.add_argument("--base-model-path", default=None, required=False,
                        help="The path of the previous trained base model to load")
    parser.add_argument("--lr", default=0.001, type=float, required=False, help="Learning rate for the model")
    parser.add_argument("--base-trainable", default="False", type=str, required=True,
                        help="If the base model is trainable")
    parser.add_argument("--use-class-weights", default="False", type=str, required=True,
                        help="Whether to use class weights to train the model")
    parser.add_argument("--description", required=True,
                        help="Describe about the model like the hyper-parameters, training params, etc. Will be helpful later to identify why the model was created")
    args = parser.parse_args()

    model_name = args.model_name
    epochs = args.epochs
    initial_epochs = args.init_epochs
    model_path = args.model_path
    base_model_path = args.base_model_path
    lr = args.lr
    if args.base_trainable == "True":
        base_trainable = True
    else:
        base_trainable = False
    if args.use_class_weights == "True":
        use_class_weights = True
    else:
        use_class_weights = False

    LOG_FILE_NAME = model_name + "_log.log"

    logger = configure_logger()
    logger.info(f"Following are the arguments passed to the script\n{args}")
    main(use_class_weights)
