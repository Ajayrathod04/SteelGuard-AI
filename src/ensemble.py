import os
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from config import MODELS_DIR, PLOTS_DIR, SEED, TARGET_COL
from utils import SimpleLogger, seed_everything, calculate_metrics
from visualization import (
    plot_precision_recall_curve, plot_confusion_matrix,
    plot_precision_recall_vs_threshold
)

def calculate_f2(y_true, y_pred):
    tp = np.sum(y_pred & (y_true == 1))
    fp = np.sum(y_pred & (y_true == 0))
    fn = np.sum(~y_pred & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (4 * precision + recall) > 0:
        f2 = 5 * precision * recall / (4 * precision + recall)
    else:
        f2 = 0.0
    return f2, precision, recall

def run_ensembling():
    logger = SimpleLogger("SteelGuard-AI-Ensemble")
    logger.info("Initializing Ultra-Safe Ensemble Optimization (Penalized F2)...")
    
    # Seed for reproducibility
    seed_everything(SEED)
    
    # Load OOF predictions
    oof_path = os.path.join(MODELS_DIR, 'oof_data.joblib')
    if not os.path.exists(oof_path):
        raise FileNotFoundError(f"OOF data not found at {oof_path}. Please run train.py first.")
        
    oof_data = joblib.load(oof_path)
    oof_predictions = oof_data['oof_predictions']
    y_true = oof_data['y_true']
    
    # Stratified 5-Fold setup to compute fold-wise metric variances
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(np.zeros(len(y_true)), y_true))
    
    logger.info("Loaded OOF predictions for models: XGBoost and CatBoost")
    
    # Grid search weight candidates: w_cat, w_xgb
    # Bounded between [0.30, 0.70] and sum to 1.0 (No model dominance > 0.70)
    weight_candidates = []
    for w_cat in np.linspace(0.30, 0.70, 41):
        w_xgb = 1.0 - w_cat
        if 0.30 - 1e-9 <= w_xgb <= 0.70 + 1e-9:
            weight_candidates.append([w_cat, w_xgb])
            
    logger.info(f"Generated {len(weight_candidates)} stable ensemble weight combinations bounded in [0.30, 0.70].")
    
    best_penalized_score = -9999.0
    best_weights = None
    best_threshold = 0.5
    best_recall = -1.0
    best_precision = -1.0
    best_f2 = -1.0
    best_recall_std = -1.0
    best_precision_std = -1.0
    best_sensitivity_penalty = -1.0
    
    # Sweep thresholds strictly in [0.045, 0.07] narrow range for safety
    thresholds_sweep = np.linspace(0.045, 0.07, 1001)
    
    # Prepare arrays for fast search
    cat_probs = oof_predictions['catboost']
    xgb_probs = oof_predictions['xgboost']
    
    for w in weight_candidates:
        w_cat, w_xgb = w
        # Blended soft voting probabilities
        y_prob_blend = w_cat * cat_probs + w_xgb * xgb_probs
        
        for t in thresholds_sweep:
            y_pred_blend = (y_prob_blend >= t).astype(int)
            
            # 1. Compute fold-wise metrics
            fold_recalls = []
            fold_precisions = []
            
            for train_idx, val_idx in fold_splits:
                y_true_fold = y_true[val_idx]
                y_pred_fold = y_pred_blend[val_idx]
                
                tp = np.sum(y_pred_fold & (y_true_fold == 1))
                fp = np.sum(y_pred_fold & (y_true_fold == 0))
                fn = np.sum(~y_pred_fold & (y_true_fold == 1))
                
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                
                fold_recalls.append(rec)
                fold_precisions.append(prec)
                
            recall_std = np.std(fold_recalls)
            precision_std = np.std(fold_precisions)
            
            # Global F2
            f2, prec, rec = calculate_f2(y_true, y_pred_blend)
            
            # 2. Compute Threshold Sensitivity Penalty: measure volatility at t +/- 0.005
            f2_minus, _, _ = calculate_f2(y_true, (y_prob_blend >= max(0.001, t - 0.005)).astype(int))
            f2_plus, _, _ = calculate_f2(y_true, (y_prob_blend >= min(0.999, t + 0.005)).astype(int))
            sensitivity_penalty = np.std([f2_minus, f2, f2_plus])
            
            # Penalized F2 Score Formula
            penalized_score = f2 - 0.5 * recall_std - 0.5 * precision_std - 1.0 * sensitivity_penalty
            
            # Tie breaking logic: prefer threshold closest to the middle of the range (0.0575)
            is_better = False
            if penalized_score > best_penalized_score + 1e-9:
                is_better = True
            elif abs(penalized_score - best_penalized_score) < 1e-9:
                if abs(t - 0.0575) < abs(best_threshold - 0.0575):
                    is_better = True
                    
            if is_better:
                best_penalized_score = penalized_score
                best_weights = w
                best_threshold = t
                best_recall = rec
                best_precision = prec
                best_f2 = f2
                best_recall_std = recall_std
                best_precision_std = precision_std
                best_sensitivity_penalty = sensitivity_penalty
                
    w_cat, w_xgb = best_weights
    logger.success("Penalized Ensemble Weight Optimization Complete!")
    logger.info(f"  Optimized Weights:")
    logger.info(f"    CatBoost     : {w_cat:.4f}")
    logger.info(f"    XGBoost      : {w_xgb:.4f}")
    logger.info(f"  Optimized Decision Threshold (t*): {best_threshold:.5f}")
    
    # Calculate full metrics on best configuration
    best_y_prob_blend = w_cat * cat_probs + w_xgb * xgb_probs
    best_y_pred_blend = (best_y_prob_blend >= best_threshold).astype(int)
    
    best_metrics = calculate_metrics(y_true, best_y_pred_blend, best_y_prob_blend)
    best_metrics['f2'] = best_f2
    best_metrics['pr_auc'] = average_precision_score(y_true, best_y_prob_blend)
    best_metrics['recall_fold_std'] = best_recall_std
    best_metrics['precision_fold_std'] = best_precision_std
    best_metrics['sensitivity_penalty'] = best_sensitivity_penalty
    
    logger.info("==================================================")
    logger.info("      STABILITY-HARDENED OOF ENSEMBLE METRICS     ")
    logger.info("==================================================")
    logger.info(f"  OOF Recall                 : {best_recall:.4f}")
    logger.info(f"  OOF Precision              : {best_precision:.4f}")
    logger.info(f"  OOF F2-Score               : {best_f2:.4f}")
    logger.info(f"  OOF PR-AUC                 : {best_metrics['pr_auc']:.4f}")
    logger.info(f"  OOF Penalized Score        : {best_penalized_score:.4f}")
    logger.info(f"  Fold Recall Std (Variance) : {best_recall_std:.4f}")
    logger.info(f"  Fold Precision Std         : {best_precision_std:.4f}")
    logger.info(f"  Threshold Sensitivity Std  : {best_sensitivity_penalty:.4f}")
    logger.info(f"  OOF False Negatives (FN)   : {best_metrics['fn']}")
    logger.info(f"  OOF True Positives (TP)    : {best_metrics['tp']}")
    logger.info(f"  OOF False Positives (FP)   : {best_metrics['fp']}")
    logger.info(f"  OOF True Negatives (TN)    : {best_metrics['tn']}")
    logger.info(f"  OOF ROC AUC                : {best_metrics['auc']:.4f}")
    logger.info("==================================================")
    
    # Save ensemble metadata
    weight_dict = {
        'catboost': float(w_cat),
        'xgboost': float(w_xgb)
    }
    
    ensemble_metadata = {
        'weights': weight_dict,
        'threshold': float(best_threshold),
        'metrics': best_metrics,
        'models_list': ['catboost', 'xgboost']
    }
    
    joblib.dump(ensemble_metadata, os.path.join(MODELS_DIR, 'ensemble_metadata.joblib'))
    joblib.dump(ensemble_metadata, os.path.join(MODELS_DIR, 'ensemble.pkl'))
    logger.success("Saved ensemble metadata to models/ensemble_metadata.joblib and models/ensemble.pkl")
    
    # Plot evaluations on OOF Blended probabilities
    pr_curve_path = os.path.join(PLOTS_DIR, 'pr_curve.png')
    plot_precision_recall_curve(y_true, best_y_prob_blend, best_threshold, pr_curve_path)
    logger.success(f"Saved Precision-Recall curve to {pr_curve_path}")
    
    cm_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, best_y_pred_blend, cm_path)
    logger.success(f"Saved Confusion Matrix to {cm_path}")
    
    thresh_search_path = os.path.join(PLOTS_DIR, 'threshold_search_graph.png')
    plot_precision_recall_vs_threshold(y_true, best_y_prob_blend, thresh_search_path, opt_threshold=best_threshold)
    logger.success(f"Saved Threshold Search Graph to {thresh_search_path}")
    
    # ==================================================
    # PHASE 5: FALSE NEGATIVE MICRO-ANALYSIS
    # ==================================================
    logger.info("PHASE 5: Inspecting OOF False Negatives (FN) for repeatable pattern discovery...")
    fn_indices = np.where((best_y_pred_blend == 0) & (y_true == 1))[0]
    
    if len(fn_indices) > 0:
        logger.info(f"Found {len(fn_indices)} False Negatives. Displaying prediction profile:")
        for idx in fn_indices[:10]:
            logger.info(f"  OOF Sample {idx:4d} | True Class: 1 | Blend Prob: {best_y_prob_blend[idx]:.5f} | CatBoost: {cat_probs[idx]:.5f} | XGBoost: {xgb_probs[idx]:.5f} | Disagreement: {abs(cat_probs[idx] - xgb_probs[idx]):.5f}")
    else:
        logger.success("Zero False Negatives on OOF predictions!")
        
if __name__ == '__main__':
    run_ensembling()
