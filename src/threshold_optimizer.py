import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from utils import SimpleLogger, calculate_metrics

def optimize_threshold(y_true, y_prob, target_precision=0.90):
    """
    Find the threshold that maximizes Recall subject to Precision >= target_precision.
    Sweeps thresholds from 0.01 to 0.99.
    """
    logger = SimpleLogger("SteelGuard-AI-ThresholdOptimizer")
    
    best_threshold = 0.5
    best_recall = -1.0
    best_precision = -1.0
    best_metrics = None
    
    # Sweep thresholds
    thresholds = np.linspace(0.01, 0.99, 990)
    
    compliant_thresholds = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(float)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        if prec >= target_precision:
            compliant_thresholds.append((t, prec, rec))
            
    if compliant_thresholds:
        # Sort compliant thresholds: first by Recall (descending), then by Precision (descending)
        # We want maximum recall, and in case of tie, higher precision is preferred.
        compliant_thresholds.sort(key=lambda x: (-x[2], -x[1]))
        best_threshold, best_precision, best_recall = compliant_thresholds[0]
        
        # Calculate full metrics
        y_pred = (y_prob >= best_threshold).astype(float)
        best_metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        logger.success(
            f"Found compliant threshold! t = {best_threshold:.4f} | "
            f"Recall = {best_recall:.4f} | Precision = {best_precision:.4f} | "
            f"False Negatives = {best_metrics['fn']}"
        )
    else:
        logger.warning(
            f"No threshold achieved Precision >= {target_precision:.2f}. "
            "Falling back to maximum F1-Score threshold."
        )
        # Fallback to max F1-Score
        best_f1 = -1.0
        for t in thresholds:
            y_pred = (y_prob >= t).astype(float)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            if prec + rec > 0:
                f1 = 2 * (prec * rec) / (prec + rec)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t
                    best_precision = prec
                    best_recall = rec
                    
        y_pred = (y_prob >= best_threshold).astype(float)
        best_metrics = calculate_metrics(y_true, y_pred, y_prob)
        
    return best_threshold, best_metrics
