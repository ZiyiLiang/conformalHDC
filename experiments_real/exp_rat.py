import sys
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append('../') 
try:
    from NeuroHDC.FHRR import RFF
    from conformalHDC.models import ConformalHDC
    from conformalHDC.utils import eval_m_psets
except ImportError:
    print("Warning: NeuroHDC or conformalHDC modules not found. Ensure '../' is in path.")
