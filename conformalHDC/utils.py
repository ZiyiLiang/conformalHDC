import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm



def eval_psets(S, y):
    coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    length = np.mean([len(S[i]) for i in range(len(y))])
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    length_cover = np.mean([len(S[i]) for i in idx_cover])
    return coverage, length, length_cover


def eval_m_psets(S,y):
    results_tmp = pd.DataFrame({})
    # Evaluate the marginal coverage
    coverage, length, length_cover = eval_psets(S, y)
    results_tmp["M-coverage"] = [coverage]
    results_tmp["M-size"] = [length]
    results_tmp["M-size|cov"] = [length_cover]
    return results_tmp


def eval_lc_psets(S,y):
    # Evaluate the label-conditional coverage
    results_tmp = pd.DataFrame({})

    for i in np.unique(y):
        label = i
        idx = np.where(y==label)[0]
        coverage, length, length_cover = eval_psets(np.array(S, dtype=object)[idx], np.array(y)[idx])

        results_tmp["class"+str(label)+"-LC-coverage"] = [coverage]
        results_tmp["class"+str(label)+"-LC-size"] = [length]
        results_tmp["class"+str(label)+"-LC-size|cov"] = [length_cover]
    return results_tmp


def eval_accuracy(y_pred, y_true):
    ''' Computes standard classification accuracy. 
        Args:
            y_pred: Array-like of predicted labels.
            y_true: Array-like of ground truth labels.
        Returns:
            Float accuracy score [0.0, 1.0].
    '''
    # Ensure inputs are numpy arrays for element-wise comparison
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    if len(y_pred) != len(y_true):
        raise ValueError(f"Shape mismatch: preds {len(y_pred)} vs true {len(y_true)}")
        
    return np.mean(y_pred == y_true)


def eval_lc_accuracy(y_pred, y_true, labels_id):
    ''' Computes label-conditional accuracy (accuracy per class).
    
        Args:
            y_pred: Array-like of predicted labels.
            y_true: Array-like of ground truth labels.
            labels_id: List of label IDs to evaluate (e.g., [0, 1, 2]).
            
        Returns:
            List of float accuracy scores corresponding to the order of labels_id.
            Returns np.nan for classes with no samples in y_true.
    '''
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    if len(y_pred) != len(y_true):
        raise ValueError(f"Shape mismatch: preds {len(y_pred)} vs true {len(y_true)}")
        
    accuracies = []
    
    for label in labels_id:
        # Mask where the ground truth is the current class
        mask = (y_true == label)
        
        if np.sum(mask) == 0:
            # Avoid division by zero if class is missing from test set
            accuracies.append(np.nan) 
        else:
            # Compute accuracy only on this subset
            acc = np.mean(y_pred[mask] == y_true[mask])
            accuracies.append(acc)
            
    return accuracies