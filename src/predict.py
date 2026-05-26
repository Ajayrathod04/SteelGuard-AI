import os
import pandas as pd
import numpy as np
import joblib

from config import DATA_DIR, MODELS_DIR, SUBMISSIONS_DIR, TARGET_COL, ID_COL, N_SPLITS, N_REPEATS
from utils import SimpleLogger, load_model

def run_predictions():
    logger = SimpleLogger("SteelGuard-AI-Prediction")
    logger.info("Initializing Elite Prediction Pipeline...")
    
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
    
    logger.info(f"Loaded ensemble metadata. Models weights: {weights}")
    logger.info(f"Loaded decision threshold: {threshold:.4f}")
    
    # Load primary feature names (original columns before FE)
    train_path = os.path.join(DATA_DIR, 'train.csv')
    df_train_raw = pd.read_csv(train_path)
    primary_features = [col for col in df_train_raw.columns if col not in [ID_COL, TARGET_COL]]
    
    # Final features expected by model (engineered features)
    final_features = joblib.load(os.path.join(MODELS_DIR, 'final_features.joblib'))
    
    # Array to accumulate probabilities for all test samples
    # We will average over 25 splits (5 folds x 5 repeats)
    accumulated_test_probs = np.zeros(len(df_test))
    
    total_splits = N_SPLITS * N_REPEATS
    logger.info(f"Performing leakage-free, fold-consistent test predictions across {total_splits} splits...")
    
    for rep in range(N_REPEATS):
        for fold in range(N_SPLITS):
            # A. Load preprocessor and feature engineer for this split
            preproc_path = os.path.join(MODELS_DIR, f"preprocessor_fold_{fold}_rep_{rep}.joblib")
            eng_path = os.path.join(MODELS_DIR, f"feature_engineer_fold_{fold}_rep_{rep}.joblib")
            
            preprocessor = load_model(preproc_path)
            engineer = load_model(eng_path)
            
            # B. Preprocess and feature engineer test data
            df_test_proc = preprocessor.transform(df_test)
            df_test_eng = engineer.transform(df_test_proc)
            
            X_test = df_test_eng[final_features].values
            
            # C. Predict probability with each model, weighted by its optimized weight
            split_prob_blend = np.zeros(len(df_test))
            
            for model_name in models_list:
                model_path = os.path.join(MODELS_DIR, f"{model_name}_fold_{fold}_rep_{rep}.joblib")
                model = load_model(model_path)
                
                # Predict probabilities
                model_probs = model.predict_proba(X_test)[:, 1]
                
                # Weighted contribution
                split_prob_blend += weights[model_name] * model_probs
                
            # Accumulate this split's predictions
            accumulated_test_probs += split_prob_blend / total_splits
            
    logger.success(f"Test predictions accumulated and averaged across {total_splits} splits!")
    
    # 3. Apply Optimized Threshold
    test_binary_predictions = (accumulated_test_probs >= threshold).astype(int)
    
    # 4. Map predictions to exact order of sample_submission
    # Create a mapping of CoilID -> Prediction
    prediction_map = dict(zip(df_test[ID_COL], test_binary_predictions))
    
    # Check if there are any missing CoilIDs in our test predictions compared to sample_submission
    missing_ids = set(df_sample_sub[ID_COL]) - set(prediction_map.keys())
    if missing_ids:
        logger.warning(f"Found {len(missing_ids)} CoilIDs in sample_submission that were missing in test.csv! Filling with 0.")
        for mid in missing_ids:
            prediction_map[mid] = 0
            
    # Construct final submission DataFrame following exact sample_submission ordering
    df_submission = pd.DataFrame({
        ID_COL: df_sample_sub[ID_COL],
        TARGET_COL: [prediction_map[cid] for cid in df_sample_sub[ID_COL]]
    })
    
    # 5. Verify constraints
    assert len(df_submission) == len(df_sample_sub), "Row count does not match sample submission!"
    assert list(df_submission.columns) == list(df_sample_sub.columns), "Columns do not match sample submission!"
    assert (df_submission[ID_COL] == df_sample_sub[ID_COL]).all(), "CoilID ordering does not match sample submission!"
    
    # 6. Save final submission
    sub_output_path = os.path.join(SUBMISSIONS_DIR, 'expected_submission.csv')
    df_submission.to_csv(sub_output_path, index=False)
    logger.success(f"Final competition submission saved to {sub_output_path}")
    
    # 7. Print prediction statistics
    predicted_counts = df_submission[TARGET_COL].value_counts()
    logger.info("Final Submission Statistics:")
    logger.info(f"  Total test samples    : {len(df_submission)}")
    logger.info(f"  Predicted Normal (0)  : {predicted_counts.get(0, 0)}")
    logger.info(f"  Predicted Defect (1)  : {predicted_counts.get(1, 0)}")
    logger.info(f"  Predicted Defect Rate : {predicted_counts.get(1, 0)/len(df_submission)*100:.2f}%")
    
if __name__ == '__main__':
    run_predictions()
