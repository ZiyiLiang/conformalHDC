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