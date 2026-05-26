import random
import os
import numpy as np
import joblib
from sklearn.metrics import recall_score, precision_score, confusion_matrix, roc_auc_score

def seed_everything(seed=42):
    """
    Set seeds for full reproducibility across python, numpy, and environment variables.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # Most tree packages use their internal parameters for random_state,
    # but setting numpy/random seeds ensures deterministic behaviors in CV, imputations, etc.

class SimpleLogger:
    """
    Clean and beautiful print logger.
    """
    def __init__(self, name="SteelGuard-AI"):
        self.name = name

    def info(self, msg):
        print(f"[*] [{self.name}] INFO: {msg}", flush=True)

    def success(self, msg):
        print(f"[+] [{self.name}] SUCCESS: {msg}", flush=True)

    def warning(self, msg):
        print(f"[!] [{self.name}] WARNING: {msg}", flush=True)

    def error(self, msg):
        print(f"[-] [{self.name}] ERROR: {msg}", flush=True)

def save_model(model, filepath):
    """
    Save a model to disk using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Load a model from disk.
    """
    return joblib.load(filepath)

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate and return a dictionary of evaluation metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Handle single class confusion matrix edge cases
    if cm.shape == (1, 1):
        tn = cm[0, 0]
        fp = fn = tp = 0
    else:
        tn, fp, fn, tp = cm.ravel()
        
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        'recall': recall,
        'precision': precision,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }
    
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.5
            
    return metrics
