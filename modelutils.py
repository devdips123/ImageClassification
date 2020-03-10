import os
import json
import numpy as np
import tensorflow as tf
keras = tf.keras
import class_weights_utils as cwutils

BASE_MODEL_NAME = 'ecom-image-model_basemodel_2020_02_20_20_25_21.h5'

def read_list_from_disk(filename, directory="."):
    """
    Reads a list stored as a file from the disk and returns it as a python list
    """
    plist = []
    if not os.path.exists(filename):
        print(f"File path not found: {filename}\n List could not be retrieved")
    with open(filename, 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            l = line[:-1]
            plist.append(l)

    return plist

def read_history(filename):
    history = {}
    """
    Reads the history object from the disk and returns it as a python dictionary
    :param: filename - full path of the history object or only the filename if in the current directory
    """
    try:
        history = open(filename).read()
        history = dict(eval(json.loads(history), {"array": np.array, "float32": float}))
    except FileNotFoundError:
        print(f"No file found at the path specified: {filename}")
        
    return history

def create_model_with_cw(class_weights, lr = 0.0001): 
    """
    Creates a model with custom weighted loss function
    Metrics attached: tp, tn, fp, fn, precision, recall, auc, accuracy
    Optimizer: adam with default lr=0.0001
    Loss Function: cwutils.get_weighted_loss(class_weights)
    
    :param: class_weights: a numpy array of class-weights for all classes
    
    :return: model
    """
    lr = 0.0001
    thresholds = [float(i/100) for i in range(0, 101)]

    # metrics
    tp = keras.metrics.TruePositives(thresholds=thresholds, name="tp")
    tn = keras.metrics.TrueNegatives(thresholds=thresholds, name="tn")
    fp = keras.metrics.FalsePositives(thresholds=thresholds, name="fp")
    fn = keras.metrics.FalseNegatives(thresholds=thresholds, name="fn")
    precision = keras.metrics.Precision(thresholds=thresholds, name='precision')
    recall = keras.metrics.Recall(thresholds=thresholds, name='recall')
    auc = keras.metrics.AUC(name="auc", thresholds=thresholds)

    # Optimizer
    adam = keras.optimizers.Adam(learning_rate=lr)

    # Loss function
    custom_loss = cwutils.get_weighted_loss(class_weights)

    # Base model
    base_model = keras.models.load_model(BASE_MODEL_NAME)
    base_model.trainable = False

    model_cw = keras.models.Sequential()
    model_cw.add(base_model)
    model_cw.add(keras.layers.Dense(len(features), activation="sigmoid"))

    model_cw.compile(optimizer=adam,
                  loss=custom_loss,
                  metrics=['accuracy', tp, tn, fp, fn, precision, recall, auc])
    return model_cw