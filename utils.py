# import os
# import csv
import numpy as np
# import pandas as pd

from sklearn.metrics import *

import torch
from torch.utils.data import DataLoader


def compute_metrics(y_true, y_pred):
    """
        Computes prediction quality metrics.

        Parameters:
        ----------
        y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) labels.

        y_pred : 1d array-like, or label indicator array / sparse matrix
            Predicted labels, as returned by a classifier.

        Returns:
        --------
        accuracy : accuracy
        conf_mat : confusion matrix
        precision : weighted precision score
        recall : weighted recall score
        f1 : weighted f1 score
    """
    accuracy  = accuracy_score(y_true, y_pred)
    conf_mat  = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall    = recall_score(y_true, y_pred, average='weighted')
    f1        = f1_score(y_true, y_pred, average='weighted')
    return accuracy, conf_mat, precision, recall, f1



def get_dataloader(dataset, batch_size=1):
    """
        Converts a dataset to a dataloader.
        
        Parameters:
        ----------
        
        X : numpy ndarray
            Input dataset with columns as features and rows as observations.

        y : numpy ndarray
            Class labels.

        batch_size: int, default=1
            The batch size.

    
        Returns:
        --------
        dataloader : a pytorch dataloader. 
    """
      
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader



def sample_vec(vec, n):
    """
        Subsample a vector uniformly from each level. Used to subsample datasets with several classes in a balanced manner.
        
        Parameters:
        ----------
        vec : numpy ndarray
            The vector to sample from.

        n : int
            Number of samples per level.

        Returns:
        --------
        to_ret : a numpy array including indices of the selected subset.
    """
    vec_list = vec.tolist()
    vec_list = set(vec_list)
    to_ret = np.array([], dtype='int')
    for val in vec_list:
        ii = np.where(vec == val)[0] 
        index = np.random.choice(ii, n)
        to_ret = np.append(to_ret, index)
    return to_ret