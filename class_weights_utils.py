from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd

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

def calculate_class_weights(df, features, classes= np.array([0., 1.]), weight_type='balanced'):
    """
    Computes the class weights for each label in the multi-label classification
    :param df: data-frame containing the data
    :param features: The list of features or the labels
    :param classes: target classes for each label. It is [0,1]
    :param type: 'balanced', None or a dict containing weights for each class
    :return: A ndarray of class-weights
    """

    num_features = len(features)
    num_classes = len(classes)
    class_weights = np.empty((num_features, num_classes))
    for i in range(num_features):
        class_weights[i] = compute_class_weight(class_weight=weight_type, classes=classes, y=df[features[i]].values)

    return class_weights