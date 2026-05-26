import os
import numpy as np
import joblib
from sklearn.metrics import average_precision_score

from config import MODELS_DIR, PLOTS_DIR, SEED
from utils import SimpleLogger, seed_everything, calculate_metrics
from threshold_optimizer import optimize_threshold
from visualization import (
    plot_precision_recall_curve, plot_confusion_matrix,
    plot_feature_importance, plot_precision_recall_vs_threshold
)

def run_ensembling():
    logger = SimpleLogger("SteelGuard-AI-Ensemble")
    logger.info("Initializing Elite Ensemble Optimization...")
    
    # Seed for reproducibility
    seed_everything(SEED)
    
    # Load OOF predictions
    oof_path = os.path.join(MODELS_DIR, 'oof_data.joblib')
    if not os.path.exists(oof_path):
        raise FileNotFoundError(f"OOF data not found at {oof_path}. Please run train.py first.")
        
    oof_data = joblib.load(oof_path)
    oof_predictions = oof_data['oof_predictions']
    y_true = oof_data['y_true']
    
    models = list(oof_predictions.keys())
    n_models = len(models)
    
    logger.info(f"Loaded OOF predictions for models: {models}")
    
    # Generate OOF prediction matrix: shape (n_samples, n_models)
    oof_matrix = np.column_stack([oof_predictions[m] for m in models])
    
    # We will search for weights using Dirichlet distribution to generate random weights on the simplex (sum to 1)
    np.random.seed(SEED)
    n_trials = 3000
    dirichlet_weights = np.random.dirichlet(alpha=np.ones(n_models), size=n_trials)
    
    # Add equal weights and single model weights as baselines
    baselines = [np.ones(n_models) / n_models]  # Equal weights
    for i in range(n_models):
        w = np.zeros(n_models)
        w[i] = 1.0
        baselines.append(w)
    
    all_weights = np.vstack([baselines, dirichlet_weights])
    
    best_weights = None
    best_threshold = 0.5
    best_recall = -1.0
    best_precision = -1.0
    
    target_precision_constraint = 0.905  # 90.5% with 0.5% safety buffer for hidden leaderboard robustness
    
    logger.info("Searching for optimal ensemble weights and threshold maximizing OOF Recall subject to Precision >= 90%...")
    
    n_pos = np.sum(y_true)
    thresholds_sweep = np.linspace(0.01, 0.99, 199)
    
    for w in all_weights:
        y_prob_blend = np.dot(oof_matrix, w)
        
        for t in thresholds_sweep:
            y_pred = (y_prob_blend >= t)
            tp = np.sum(y_pred & (y_true == 1))
            fp = np.sum(y_pred & (y_true == 0))
            
            total_pred = tp + fp
            if total_pred > 0:
                prec = tp / total_pred
                rec = tp / n_pos
                
                if prec >= target_precision_constraint:
                    if rec > best_recall:
                        best_recall = rec
                        best_precision = prec
                        best_weights = w
                        best_threshold = t
                    elif rec == best_recall and prec > best_precision:
                        best_precision = prec
                        best_weights = w
                        best_threshold = t
                        
    # Fallback if no weight blend achieves Precision >= target_precision_constraint
    if best_weights is None:
        logger.warning("No weight blend achieved Precision >= 90.5%. Falling back to PR-AUC maximization.")
        best_pr_auc = -1.0
        for w in all_weights:
            y_prob_blend = np.dot(oof_matrix, w)
            pr_auc = average_precision_score(y_true, y_prob_blend)
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                best_weights = w
        
        logger.info("Optimizing decision threshold on the fallback ensemble blend...")
        best_y_prob_blend = np.dot(oof_matrix, best_weights)
        best_threshold, best_metrics = optimize_threshold(y_true, best_y_prob_blend, target_precision=target_precision_constraint)
    else:
        logger.success("Found a precision-compliant ensembled configuration!")
        best_y_prob_blend = np.dot(oof_matrix, best_weights)
        best_threshold, best_metrics = optimize_threshold(y_true, best_y_prob_blend, target_precision=target_precision_constraint)
    
    # Log the best ensemble configuration
    weight_dict = {models[i]: float(best_weights[i]) for i in range(n_models)}
    logger.info("Optimized Ensemble Weights:")
    for model_name, weight in weight_dict.items():
        logger.info(f"  {model_name:<15}: {weight:.4f}")
        
    logger.info(f"Optimized Threshold t*     : {best_threshold:.4f}")
    logger.info(f"OOF Recall                 : {best_metrics['recall']:.4f}")
    logger.info(f"OOF Precision              : {best_metrics['precision']:.4f}")
    logger.info(f"OOF False Negatives (FN)   : {best_metrics['fn']}")
    logger.info(f"OOF True Positives (TP)    : {best_metrics['tp']}")
    logger.info(f"OOF False Positives (FP)   : {best_metrics['fp']}")
    logger.info(f"OOF True Negatives (TN)    : {best_metrics['tn']}")
    logger.info(f"OOF ROC AUC                : {best_metrics['auc']:.4f}")
    
    # Save ensemble metadata
    ensemble_metadata = {
        'weights': weight_dict,
        'threshold': float(best_threshold),
        'metrics': best_metrics,
        'models_list': models
    }
    joblib.dump(ensemble_metadata, os.path.join(MODELS_DIR, 'ensemble_metadata.joblib'))
    joblib.dump(ensemble_metadata, os.path.join(MODELS_DIR, 'ensemble.pkl'))
    logger.success("Saved ensemble metadata to models/ensemble_metadata.joblib and models/ensemble.pkl")
    
    # Plot evaluations on OOF Blended probabilities
    best_y_prob_blend = np.dot(oof_matrix, best_weights)
    best_y_pred_blend = (best_y_prob_blend >= best_threshold).astype(float)
    
    pr_curve_path = os.path.join(PLOTS_DIR, 'pr_curve.png')
    plot_precision_recall_curve(y_true, best_y_prob_blend, best_threshold, pr_curve_path)
    logger.success(f"Saved Precision-Recall curve to {pr_curve_path}")
    
    cm_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, best_y_pred_blend, cm_path)
    logger.success(f"Saved Confusion Matrix to {cm_path}")

    # Plot threshold search graph
    thresh_search_path = os.path.join(PLOTS_DIR, 'threshold_search_graph.png')
    plot_precision_recall_vs_threshold(y_true, best_y_prob_blend, thresh_search_path, opt_threshold=best_threshold)
    logger.success(f"Saved Threshold Search Graph to {thresh_search_path}")

    # Plot feature importances
    try:
        final_features = joblib.load(os.path.join(MODELS_DIR, 'final_features.joblib'))
        lgb_full = joblib.load(os.path.join(MODELS_DIR, 'lightgbm.pkl'))
        feat_imp_path = os.path.join(PLOTS_DIR, 'feature_importance.png')
        plot_feature_importance(lgb_full, final_features, feat_imp_path)
        logger.success(f"Saved Feature Importance plot to {feat_imp_path}")
    except Exception as e:
        logger.warning(f"Could not generate feature importance plot: {str(e)}")

if __name__ == '__main__':
    run_ensembling()
