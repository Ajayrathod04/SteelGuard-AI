import os
import pandas as pd
import numpy as np
import joblib

from config import DATA_DIR, MODELS_DIR, SUBMISSIONS_DIR, TARGET_COL, ID_COL, N_SPLITS
from utils import SimpleLogger, load_model

def run_predictions():
    logger = SimpleLogger("SteelGuard-AI-Prediction")
    logger.info("Initializing Stability-Hardened Inference Pipeline...")
    
    # 1. Load Test Data
    test_path = os.path.join(DATA_DIR, 'test.csv')
    df_test = pd.read_csv(test_path)
    logger.success(f"Successfully loaded test data: {df_test.shape}")
    
    # Load sample submission
    sample_sub_path = os.path.join(DATA_DIR, 'sample_submission.csv')
    df_sample_sub = pd.read_csv(sample_sub_path)
    logger.success(f"Successfully loaded sample submission: {df_sample_sub.shape}")
    
    # 2. Load Ensemble Metadata
    ensemble_path = os.path.join(MODELS_DIR, 'ensemble_metadata.joblib')
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(f"Ensemble metadata not found at {ensemble_path}. Please run ensemble.py first.")
        
    metadata = joblib.load(ensemble_path)
    weights = metadata['weights']
    threshold = metadata['threshold']
    models_list = metadata['models_list']
    oof_metrics = metadata['metrics']
    
    logger.info(f"Loaded stability-hardened ensemble metadata.")
    logger.info(f"  Optimized Weights: {weights}")
    logger.info(f"  OOF Optimized Penalized Threshold: {threshold:.5f}")
    
    # Array to accumulate probabilities for all test samples
    accumulated_test_probs = np.zeros(len(df_test))
    
    logger.info(f"Performing leak-free fold-bagged predictions with adaptive feature selection across {N_SPLITS} splits...")
    
    for fold in range(N_SPLITS):
        # A. Load preprocessor, feature engineer, and selected features list for this fold
        preproc_path = os.path.join(MODELS_DIR, f"preprocessor_fold_{fold}_rep_0.joblib")
        eng_path = os.path.join(MODELS_DIR, f"feature_engineer_fold_{fold}_rep_0.joblib")
        mask_path = os.path.join(MODELS_DIR, f"selected_features_fold_{fold}.joblib")
        
        preprocessor = load_model(preproc_path)
        engineer = load_model(eng_path)
        selected_features = joblib.load(mask_path)
        
        # B. Preprocess and feature engineer test data
        df_test_proc = preprocessor.transform(df_test)
        df_test_eng = engineer.transform(df_test_proc)
        
        # Filter features to keep only top 85% cumulative importance of this fold
        X_test_sel = df_test_eng[selected_features].values
        
        # C. Predict probability with each model, weighted by its optimized weight
        split_prob_blend = np.zeros(len(df_test))
        
        for model_name in models_list:
            model_path = os.path.join(MODELS_DIR, f"{model_name}_fold_{fold}_rep_0.joblib")
            model = load_model(model_path)
            
            # Predict probabilities
            model_probs = model.predict_proba(X_test_sel)[:, 1]
            
            # Weighted contribution
            split_prob_blend += weights[model_name] * model_probs
            
        # Accumulate this split's predictions
        accumulated_test_probs += split_prob_blend / N_SPLITS
        
    logger.success(f"Test predictions ensembled and bagged across {N_SPLITS} splits!")
    
    # 3. Apply Optimized Threshold (No hard positive capping/clipping)
    test_binary_predictions = (accumulated_test_probs >= threshold).astype(int)
    
    # 4. Construct final submission DataFrame following exact test.csv ordering
    df_submission = pd.DataFrame({
        ID_COL: df_test[ID_COL],
        TARGET_COL: test_binary_predictions
    })
    
    # 5. Automated Validation & Constraints Check
    if len(df_submission) != 339:
        raise ValueError(f"CRITICAL ERROR: Expected exactly 339 rows in submission, but generated {len(df_submission)}.")
    
    expected_cols = [ID_COL, TARGET_COL]
    if list(df_submission.columns) != expected_cols:
        raise ValueError(f"CRITICAL ERROR: Columns do not match expected schema {expected_cols}. Got {list(df_submission.columns)}.")
        
    if df_submission.isnull().any().any():
        raise ValueError("CRITICAL ERROR: Submission contains null values!")
        
    unique_preds = set(df_submission[TARGET_COL].unique())
    if not unique_preds.issubset({0, 1}):
        raise ValueError(f"CRITICAL ERROR: Invalid prediction values found: {unique_preds}. Must only contain 0 or 1.")
        
    logger.success("Submission validation passed:")
    logger.success("  [OK] Rows = 339")
    logger.success("  [OK] Columns correct")
    logger.success("  [OK] No null values")
    logger.success("  [OK] Binary predictions only")
    
    # 6. Save final submission (Force update to both paths for VS Code caching)
    sub_output_path = os.path.join(SUBMISSIONS_DIR, 'expected_submission.csv')
    df_submission.to_csv(sub_output_path, index=False)
    logger.success(f"Final competition submission saved to {sub_output_path}")
    
    # Force save to both D: and d: to reload open tabs in VS Code
    for drive in ['d', 'D']:
        alt_path = sub_output_path.replace('d:', f'{drive}:').replace('D:', f'{drive}:')
        try:
            os.makedirs(os.path.dirname(alt_path), exist_ok=True)
            df_submission.to_csv(alt_path, index=False)
            logger.info(f"Synchronized submission to: {alt_path}")
        except Exception:
            pass
            
    # 7. Print prediction and optimization statistics
    predicted_counts = df_submission[TARGET_COL].value_counts()
    predicted_pos = int(predicted_counts.get(1, 0))
    logger.info("==================================================")
    logger.info("      FINAL TEST SUBMISSION PROFILE & METRICS     ")
    logger.info("==================================================")
    logger.info(f"  Applied Decision Threshold : {threshold:.5f}")
    logger.info(f"  OOF Validation Recall      : {oof_metrics['recall']:.4f}")
    logger.info(f"  OOF Validation Precision   : {oof_metrics['precision']:.4f}")
    logger.info(f"  OOF Validation F2-Score    : {oof_metrics['f2']:.4f}")
    logger.info(f"  OOF Validation PR-AUC      : {oof_metrics['pr_auc']:.4f}")
    logger.info(f"  OOF False Negatives (FN)   : {oof_metrics['fn']}")
    logger.info(f"  OOF False Positives (FP)   : {oof_metrics['fp']}")
    logger.info(f"  Total Test Samples         : {len(df_submission)}")
    logger.info(f"  Predicted Normal (0)       : {predicted_counts.get(0, 0)}")
    logger.info(f"  Predicted Defect (1)       : {predicted_pos}")
    logger.info(f"  Defect Rate                : {predicted_pos/len(df_submission)*100:.2f}%")
    
    # Soft distribution monitoring against healthy range [35, 70]
    if 35 <= predicted_pos <= 70:
        logger.success(f"[HEALTHY] predicted defect count ({predicted_pos}) lies within the preferred [35, 70] range.")
    else:
        logger.warning(f"[MONITOR] predicted defect count ({predicted_pos}) is outside the preferred [35, 70] range (Generalization adaptive).")
    logger.info("==================================================")

if __name__ == '__main__':
    run_predictions()
