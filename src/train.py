import os
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import joblib

from config import (
    BASE_DIR, DATA_DIR, MODELS_DIR, TARGET_COL, ID_COL, SEED, N_SPLITS, N_REPEATS,
    XGB_PARAMS, LGB_PARAMS, CAT_PARAMS, RF_PARAMS, ET_PARAMS
)
from utils import seed_everything, SimpleLogger, save_model, calculate_metrics
from preprocessing import RobustPreprocessor
from feature_engineering import LeakageFreeFeatureEngineer
from visualization import plot_shap_summary

def run_training():
    logger = SimpleLogger("SteelGuard-AI-Training")
    logger.info("Initializing Elite Training Pipeline...")
    
    # 1. Seed everything for reproducibility
    seed_everything(SEED)
    
    # 2. Load Train Data
    train_path = os.path.join(DATA_DIR, 'train.csv')
    df_train = pd.read_csv(train_path)
    logger.success(f"Successfully loaded training data: {df_train.shape}")
    
    # Define primary features
    feature_cols = [col for col in df_train.columns if col not in [ID_COL, TARGET_COL]]
    
    # Identify missingness columns globally across the entire training dataset
    global_nan_counts = df_train[feature_cols].isnull().sum()
    global_columns_with_nan = list(global_nan_counts[global_nan_counts > 0].index)
    logger.info(f"Identified {len(global_columns_with_nan)} global columns with missing values: {global_columns_with_nan}")
    
    # 3. Initialize CV
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    
    # 4. Prepare OOF arrays
    # We will track OOF predicted probabilities for each model
    models_to_train = ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'extra_trees']
    oof_predictions = {model: np.zeros(len(df_train)) for model in models_to_train}
    
    # Maintain lists to store validation fold indices and ground truths to double-check shapes
    y_true_all = df_train[TARGET_COL].values
    
    total_folds = N_SPLITS * N_REPEATS
    logger.info(f"Starting {N_SPLITS}-Fold Repeated Stratified CV ({N_REPEATS} repeats, total {total_folds} folds)...")
    
    # Loop over CV splits
    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(df_train, df_train[TARGET_COL])):
        rep = fold_idx // N_SPLITS
        fold = fold_idx % N_SPLITS
        
        logger.info(f"--- Training Repeat {rep+1}/{N_REPEATS} | Fold {fold+1}/{N_SPLITS} (Overall Fold Index: {fold_idx+1}/{total_folds}) ---")
        
        # Split features and target
        df_train_fold = df_train.iloc[train_idx].copy()
        df_val_fold = df_train.iloc[val_idx].copy()
        
        y_train = df_train_fold[TARGET_COL].values
        y_val = df_val_fold[TARGET_COL].values
        
        # A. Preprocessing
        preprocessor = RobustPreprocessor(scale_features=True, columns_with_nan=global_columns_with_nan)
        df_train_fold_proc = preprocessor.fit_transform(df_train_fold, feature_cols)
        df_val_fold_proc = preprocessor.transform(df_val_fold)
        
        # Save preprocessor for this specific fold/repeat
        preproc_path = os.path.join(MODELS_DIR, f"preprocessor_fold_{fold}_rep_{rep}.joblib")
        save_model(preprocessor, preproc_path)
        
        # Identify columns generated after preprocessing (like isnull columns)
        proc_feature_cols = [c for c in df_train_fold_proc.columns if c not in [ID_COL, TARGET_COL]]
        
        # B. Feature Engineering
        engineer = LeakageFreeFeatureEngineer(seed=SEED)
        df_train_fold_eng = engineer.fit_transform(df_train_fold_proc, proc_feature_cols)
        df_val_fold_eng = engineer.transform(df_val_fold_proc)
        
        # Save feature engineer for this specific fold/repeat
        eng_path = os.path.join(MODELS_DIR, f"feature_engineer_fold_{fold}_rep_{rep}.joblib")
        save_model(engineer, eng_path)
        
        # Identify final set of engineered feature columns
        final_feature_cols = [c for c in df_train_fold_eng.columns if c not in [ID_COL, TARGET_COL]]
        
        X_train = df_train_fold_eng[final_feature_cols].values
        X_val = df_val_fold_eng[final_feature_cols].values
        
        # C. Model Fitting
        
        # --- Model 1: XGBoost ---
        xgb_clf = XGBClassifier(**XGB_PARAMS)
        xgb_clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        oof_predictions['xgboost'][val_idx] += xgb_clf.predict_proba(X_val)[:, 1] / N_REPEATS
        save_model(xgb_clf, os.path.join(MODELS_DIR, f"xgboost_fold_{fold}_rep_{rep}.joblib"))
        
        # --- Model 2: LightGBM ---
        lgb_clf = LGBMClassifier(**LGB_PARAMS)
        lgb_clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)]
        )
        oof_predictions['lightgbm'][val_idx] += lgb_clf.predict_proba(X_val)[:, 1] / N_REPEATS
        save_model(lgb_clf, os.path.join(MODELS_DIR, f"lightgbm_fold_{fold}_rep_{rep}.joblib"))
        
        # --- Model 3: CatBoost ---
        cat_clf = CatBoostClassifier(**CAT_PARAMS)
        cat_clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            use_best_model=True
        )
        oof_predictions['catboost'][val_idx] += cat_clf.predict_proba(X_val)[:, 1] / N_REPEATS
        save_model(cat_clf, os.path.join(MODELS_DIR, f"catboost_fold_{fold}_rep_{rep}.joblib"))
        
        # --- Model 4: Random Forest ---
        rf_clf = RandomForestClassifier(**RF_PARAMS)
        rf_clf.fit(X_train, y_train)
        oof_predictions['random_forest'][val_idx] += rf_clf.predict_proba(X_val)[:, 1] / N_REPEATS
        save_model(rf_clf, os.path.join(MODELS_DIR, f"random_forest_fold_{fold}_rep_{rep}.joblib"))
        
        # --- Model 5: Extra Trees ---
        et_clf = ExtraTreesClassifier(**ET_PARAMS)
        et_clf.fit(X_train, y_train)
        oof_predictions['extra_trees'][val_idx] += et_clf.predict_proba(X_val)[:, 1] / N_REPEATS
        save_model(et_clf, os.path.join(MODELS_DIR, f"extra_trees_fold_{fold}_rep_{rep}.joblib"))
        
        # Run SHAP explanation on the last split using LightGBM as it represents tree models nicely
        if fold_idx == (N_SPLITS * N_REPEATS - 1):
            logger.info("Generating SHAP feature explanations on final split...")
            try:
                # Prepare a beautiful pandas df with proper column names for SHAP
                df_val_final_feat = pd.DataFrame(X_val, columns=final_feature_cols)
                plot_shap_summary(lgb_clf, df_val_final_feat, os.path.join(BASE_DIR, 'plots', 'shap_summary.png'))
                logger.success("SHAP explanation plot saved to plots/shap_summary.png")
            except Exception as e:
                logger.warning(f"Could not generate SHAP plots: {str(e)}")

    logger.success("All 25 CV splits completed!")
    
    # Save feature names for test prediction
    joblib.dump(final_feature_cols, os.path.join(MODELS_DIR, 'final_features.joblib'))
    
    # 5. Evaluate individual models using standard threshold 0.5
    logger.info("Evaluating individual models with standard threshold = 0.5 (for baseline context):")
    for name, probs in oof_predictions.items():
        preds = (probs >= 0.5).astype(float)
        m = calculate_metrics(y_true_all, preds, probs)
        logger.info(f"Model: {name.upper():<15} | AUC: {m['auc']:.4f} | Recall: {m['recall']:.4f} | Precision: {m['precision']:.4f} | FN: {m['fn']}")

    # Save OOF predictions and targets for ensembling and threshold tuning
    oof_data = {
        'oof_predictions': oof_predictions,
        'y_true': y_true_all
    }
    joblib.dump(oof_data, os.path.join(MODELS_DIR, 'oof_data.joblib'))
    logger.success("Saved out-of-fold predictions to models/oof_data.joblib")
    
    # 6. Fit Final Models on Entire Training Dataset for Production Deployment
    logger.info("Fitting Final Production-Grade Models on complete train dataset...")
    
    # Preprocessing
    full_preprocessor = RobustPreprocessor(scale_features=True, columns_with_nan=global_columns_with_nan)
    df_train_proc = full_preprocessor.fit_transform(df_train, feature_cols)
    save_model(full_preprocessor, os.path.join(MODELS_DIR, "preprocessor_full.joblib"))
    save_model(full_preprocessor, os.path.join(MODELS_DIR, "preprocessor_full.pkl"))
    
    proc_feature_cols = [c for c in df_train_proc.columns if c not in [ID_COL, TARGET_COL]]
    
    # Feature Engineering
    full_engineer = LeakageFreeFeatureEngineer(seed=SEED)
    df_train_eng = full_engineer.fit_transform(df_train_proc, proc_feature_cols)
    save_model(full_engineer, os.path.join(MODELS_DIR, "feature_engineer_full.joblib"))
    save_model(full_engineer, os.path.join(MODELS_DIR, "feature_engineer_full.pkl"))
    
    final_feature_cols = [c for c in df_train_eng.columns if c not in [ID_COL, TARGET_COL]]
    X_full = df_train_eng[final_feature_cols].values
    y_full = df_train[TARGET_COL].values
    
    # XGBoost
    xgb_params_full = XGB_PARAMS.copy()
    xgb_params_full.pop('early_stopping_rounds', None)
    xgb_full = XGBClassifier(**xgb_params_full)
    xgb_full.fit(X_full, y_full, verbose=False)
    save_model(xgb_full, os.path.join(MODELS_DIR, "xgboost.pkl"))
    save_model(xgb_full, os.path.join(MODELS_DIR, "xgboost.joblib"))
    logger.success("Fitted and saved xgboost.pkl")
    
    # LightGBM
    lgb_full = LGBMClassifier(**LGB_PARAMS)
    lgb_full.fit(X_full, y_full)
    save_model(lgb_full, os.path.join(MODELS_DIR, "lightgbm.pkl"))
    save_model(lgb_full, os.path.join(MODELS_DIR, "lightgbm.joblib"))
    logger.success("Fitted and saved lightgbm.pkl")
    
    # CatBoost
    cat_params_full = CAT_PARAMS.copy()
    cat_params_full.pop('early_stopping_rounds', None)
    cat_full = CatBoostClassifier(**cat_params_full)
    cat_full.fit(X_full, y_full, verbose=0)
    save_model(cat_full, os.path.join(MODELS_DIR, "catboost.pkl"))
    save_model(cat_full, os.path.join(MODELS_DIR, "catboost.joblib"))
    logger.success("Fitted and saved catboost.pkl")
    
    # Random Forest
    rf_full = RandomForestClassifier(**RF_PARAMS)
    rf_full.fit(X_full, y_full)
    save_model(rf_full, os.path.join(MODELS_DIR, "random_forest.pkl"))
    logger.success("Fitted and saved random_forest.pkl")
    
    # Extra Trees
    et_full = ExtraTreesClassifier(**ET_PARAMS)
    et_full.fit(X_full, y_full)
    save_model(et_full, os.path.join(MODELS_DIR, "extra_trees.pkl"))
    logger.success("Fitted and saved extra_trees.pkl")
    
    logger.success("All final production-grade models fitted and saved in models/ directory!")
    
if __name__ == '__main__':
    run_training()
