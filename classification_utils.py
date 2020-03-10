import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.metrics import classification_report

color_list  = ['red','blue','green','yellow','purple','pink','black','orange', 'magenta','cyan','violet','turquoise']

def pre_rec(tup, remove_nans=True):
    """
    Returns a tuple of (precision, recall) for a given input of (tp, tn, fp, fn)
    @param tup - A tuple of (tp, tn, fp, fn)
    """
    tp, tn, fp, fn = tup
    precision = 0.0
    recall = 0.0
    try:
        precision = tp/(tp+fp)
    except ZeroDivisionError:
        if remove_nans:
            precision = 0.0
        else:
            precision = np.nan
    
    try:
        recall = tp/(tp+fn)
    except ZeroDivisionError:
        if remove_nans:
            recall = 0.0
        else:
            recall = np.nan
        
    return (precision, recall)

def categorize_actual_preds(actuals, preds, threshold=0.5):
    """
    Find the tp, tn, fp, fn for a model with a given threshold
    @param actuals
    @param preds
    @param threshold - default = 0.5
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    #print(actuals.shape)
    if actuals.size != preds.size:
        print("Shapes of actuals and preds not equal")
        return tp, tn, fp, fn
    if np.count_nonzero(actuals) == 0:
        #print("[WARNING] No positive predictions")
        return tp, tn, fp, fn
    
    if actuals.ndim == 2:
        actuals = np.squeeze(actuals, axis=0)
        preds = np.squeeze(preds, axis=0)

    true_indices = []
    false_indices = []
    for i, val in enumerate(actuals):
        if val:
            true_indices.append(i)
        else:
            false_indices.append(i)

    # Calculate tp and fn
    for i in true_indices:
        act = actuals[i]
        pred = preds[i]
        if pred >= threshold:
            tp += 1
        else:
            fn += 1

    # Calculate tn and fp
    for i in false_indices:
        pred = preds[i]
        if pred >= threshold:
            fp += 1
        else:
            tn += 1

    return (tp, tn, fp, fn)

def roc_curve_for_model(actuals, preds):
    plt.figure(figsize=(12,12))
    #for i in range(20):
    fpr, tpr, thres = roc_curve(actuals.flatten(), preds.flatten(), drop_intermediate=False)
    plt.plot(fpr, tpr, marker='.',label="ROC")
    #plt.plot(thres, marker="o")
    plt.plot([0, 1], ls="--", label="No Skill")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend()
    plt.show()
    
def roc_comparison(model_preds, actuals, model_names=[]):
    """
    Compares the ROC for several models
    :param model_preds: A list of predicted values of several models
    :param actuals: The true_labels which is common for all models since the models are predicted on same dataset
    
    :return: a comparison plot
    """
    plt.figure(figsize=(12,12))
    for i, preds in enumerate(model_preds):
        fpr, tpr, thres = roc_curve(actuals.flatten(), preds.flatten(), drop_intermediate=False)
        auc_ = auc(fpr,tpr)
        if model_names:
            label = model_names[i]
        else:
            label = "ROC_model_"+str(i+1)
        label = f"{label} (area = {auc_:0.4f})"
        plt.plot(fpr, tpr,label=label)
    #plt.plot(thres, marker="o")
    plt.plot([0, 1], ls="--", label="No Skill")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend()
    plt.show()
    
def plot_roc_for_feature_id(ids, actuals, preds, feature_id_to_name_map=None):
    """
    Plots the roc graph for multiple input feature ids
    @param ids: int, tuple or list of feature ids
    @actuals: list of true labels
    @preds: list of predicted labels
    @feature_id_to_map_map: dictionary of feature_id to feature_name mapping
    """
    loop = []
    if isinstance(ids, int):
        loop = range(ids)
    elif isinstance(ids, tuple):
        low = min(ids[0], ids[1])
        high = max(ids[0], ids[1])
        loop = range(low, high)
    elif isinstance(ids, list):
        loop = ids
    plt.figure(figsize=(12,12))
    for i in loop:
        fpr, tpr, thres = roc_curve(actuals[:,i], preds[:,i], drop_intermediate=False)
        if feature_id_to_name_map:
            label=feature_id_to_name_map.get(i)+"("+str(i)+")"
        else:
            label = str(i)
        plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], ls="--", label="No Skills")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.show()
    
def result_classifications(actuals, preds, threshold=0.5):
    """
    @return: returns a dict of feature_id to a tuple of (tp, tn, fp, fn)
    """
    classifications = dict()
    #true_labels = test_gen.labels
    for i in range(preds.shape[1]):
        classifications[i] = categorize_actual_preds(actuals[:,i], preds[:,i], threshold=threshold)
        
    return classifications

def calculate_pres_recall(actuals, preds, classifications=None, threshold=0.5):
    """
     Calculates the precision and recall values of each label in the input dictionary for a given threshold
     The default threshold is 0.5
     
     :return : Returns a dictionary containing {label: (precision,recall)}
    """
    precision_recall_report = dict()
    if not classifications:
        classifications = result_classifications(actuals, preds, threshold=threshold)
    for i in classifications.keys():
        precision_recall_report[i] = pre_rec(classifications[i])
        
    return precision_recall_report

def calculate_f1score(prec, recall):
    f1=0.0
    if any([np.isnan(prec), np.isnan(recall)]):
        return f1
    try:
        f1 = 2*prec*recall/(prec+recall)
    except ZeroDivisionError:
        f1=0.0
        
    return f1

def plot_precision_recall_intersection(history_dict, metrics=['val_precision', 'val_recall'], color_list=color_list, epoch=-1):
    """
    Plots the precision and recall curves for each of the models passed as input. The purpose of this plot
    is to check at what threshold does the precision and recall converse and which model has the best 
    performance
    :param: history_dict - A dictionary with {model_name: history_object}
    :param: metrics - list of metrics to compare. Default: ['val_precision', 'val_recall']
    :param: color_list - A optional color list to display each model
    :param: epoch - The epoch for which to plot the metrics
    """
    
    x_ticks_labels = [float(i/100) for i in range(0,101,10)]
    x_ticks = [i for i in range(0,101,10)]
    plt.figure(figsize=(15,10))
    for i, model in enumerate(history_dict):
        history = history_dict[model]
   
        plt.plot(history[metrics[0]][epoch], label=f"{model}_{metrics[0]}", color=color_list[i], linestyle='-.')
        plt.plot(history[metrics[1]][epoch], label=f"{model}_{metrics[1]}", color=color_list[i], linestyle='--')
   
    plt.plot([50,50], [0,1], linestyle='--', label="0.5 Threshold")
    plt.grid()
    plt.xlabel("Threshold")
    plt.legend()
    plt.xticks(x_ticks,labels=x_ticks_labels)
    plt.title(f"Precision and Recall for all models - Epoch Number: {epoch}")
    plt.show()


def plot_metrics_for_models(history_list, model_names, metric_name, color_list=color_list):
    """
    Plots the corresponding training and validation data for the metric_name provided for the list of models
    and their corresponding history provided in form of list
    Works for those metrics whose X axis is always epoch
    For threshold based metrics use compare_precision_recall_curves()
    :param history_list: list of history dictionary object for models
    :param model_names: list of model names
    :color_list: list of colors to use for different models (default colors: ['red','blue','green','yellow','purple','pink'])
    :return: Plots a line plot with training attributes as solid line and validation attributes as dotted lines
    """

    plt.figure(figsize=(10,10))
    epochs = len(history_list[0]['loss'])
    xticks = [i for i in range(0, int(epochs)+1, 2)]
    for i, history in enumerate(history_list):
        #for metric_name in metric_list:
            #metric_name = metric_list[i]
        val_metric_name = "val_"+metric_name
        color = color_list[i]
        plt.plot(history[metric_name], label=model_names[i]+"_"+metric_name, color=color)
        plt.plot(history[val_metric_name], label=model_names[i]+"_"+val_metric_name, linestyle='--', color=color)
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.xticks(xticks,labels=xticks)
    plt.title(f"Training and Validation {metric_name} for different models")
    plt.legend()
    plt.show()
    
def compare_precision_recall_curves(models_dict, metric_name, epoch=-1):
    """
    Compares the precision or recall curves for different models. The precision and recall values should be an ndarray
    consisting of values corresponding to each threshold between [0, 1] in steps of 0.05
    
    :param: Dictionary of {"model_name: [np.ndarray]}
    :param: Name of the metric (either precision or recall
    
    :returns: a matplotlib plot
    """
    x_ticks_labels = [float(i/100) for i in range(0,101,10)]
    x_ticks = [i for i in range(0,101,10)]
    plt.figure(figsize=(10,10))
    for key in models_dict.keys():
        plt.plot(models_dict[key][epoch], marker='.', label=key)
    
    plt.legend()
    plt.grid()
    plt.xticks(x_ticks, x_ticks_labels)
    plt.xlabel("Threshold")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} (epoch {epoch}) comparison for different models")
    plt.show()
    
def plot_metrics_curve_all_epochs(history, metric, epochs=[]):
    """
    Plots the precison or recall curve for all epochs over a range of threshold
    :param: history - history object
    :param: metric - metric name like val_precision or val_recall
    :param: epochs - optional list of epochs
    """
    x_ticks_labels = [float(i/100) for i in range(0,101,10)]
    x_ticks = [i for i in range(0,101,10)]
    plt.figure(figsize=(10,10))
    if not epochs:
        for i, met in enumerate(history[metric]):
            val_auc = history['val_auc'][i]
            plt.plot(met, marker='.', label=f"epoch {i+1} (auc={val_auc})")
    else:
        for i in epochs:
            val_auc = history['val_auc'][i-1]
            plt.plot(history[metric][i-1], marker='.', label=f"epoch {i} (auc={val_auc})")

    plt.legend()
    plt.xticks(x_ticks, x_ticks_labels)
    plt.xlabel("Threshold")
    plt.grid()
    plt.ylabel(metric)
    plt.title(f"{metric} curve for all epochs")
    plt.show()
    
def plot_roc_curve_for_model(history, epochs=[], data="validation"):
    """
    Plots the ROC curve of model training by calculating the TPRs and FPRs by data inside the history of the
    trained model
    :param: history - A history object stored during model training
    :param: epochs - An optional list of epochs
    :param: data - either "training" or validation
    """
    
    prefix = ""
    if data == "validation":
        prefix = "val_"
    tprs = []
    fprs = []
    if len(epochs) == 0:
        epochs = [i for i in range(len(history_alllayers_imnet_d3['loss']))]
    else:
        # user enters epoch numbers starting from 1 as shown in the training screen, however the
        # epoch array starts from index 0
        epochs = [i-1 for i in epochs]
    for epoch in epochs:
        tp = history[prefix+'tp'][epoch]
        fp = history[prefix+'fp'][epoch]
        fn = history[prefix+'fn'][epoch]
        tn = history[prefix+'tn'][epoch]
        tpr = np.nan_to_num((tp)/(tp+fn))
        fpr = np.nan_to_num((fp)/(fp+tn))
        sorted_args = np.argsort(fpr)
        fprs.append(fpr[sorted_args])
        tprs.append(tpr[sorted_args])
    plt.figure(figsize=(12,12))
    for epoch in range(len(tprs)):
        auc = history[prefix+'auc'][epochs[epoch]]
        plt.plot(fprs[epoch],tprs[epoch], marker='.', label=f"Epoch {epochs[epoch]+1} (AUC={auc})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title('ROC for training Data')
    plt.show()