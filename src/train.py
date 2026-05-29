import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.stats import pearsonr
import joblib

from config import (
    BASE_DIR, DATA_DIR, MODELS_DIR, TARGET_COL, ID_COL, SEED, N_SPLITS
)
from utils import seed_everything, SimpleLogger, save_model, calculate_metrics
from preprocessing import RobustPreprocessor
from feature_engineering import LeakageFreeFeatureEngineer

def run_training():
    logger = SimpleLogger("SteelGuard-AI-Training")
    logger.info("Initializing Stability-Hardened Training Pipeline...")
    
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
    
    # 3. Setup Micro-Tuned Hyperparameters for ultra-safe regularization
    xgb_params_current = {
        'n_estimators': 400,
        'learning_rate': 0.03,
        'max_depth': 3,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'min_child_weight': 2,
        'gamma': 0.1,
        'reg_alpha': 1.5,
        'reg_lambda': 6.0,
        'random_state': SEED,
        'tree_method': 'hist',
        'n_jobs': -1
    }
    
    cat_params_current = {
        'iterations': 400,
        'learning_rate': 0.025,
        'depth': 4,
        'l2_leaf_reg': 6.0,
        'random_strength': 0.2,
        'bagging_temperature': 0.2,
        'random_seed': SEED,
        'verbose': 0
    }
    
    logger.info("Micro-Tuned Hyperparameters:")
    logger.info(f"  XGBoost: max_depth={xgb_params_current.get('max_depth')}, lr={xgb_params_current.get('learning_rate')}, min_child_weight={xgb_params_current.get('min_child_weight')}")
    logger.info(f"  CatBoost: depth={cat_params_current.get('depth')}, lr={cat_params_current.get('learning_rate')}, l2_leaf_reg={cat_params_current.get('l2_leaf_reg')}")

    # Protected features set that can never be pruned
    protected_set = {
        'X15', 'X30', 'X31', 'X32', 'X33', 'X34', 'X35',
        'X15_div_X30', 'X15_minus_X30', 'X31_mul_X32', 'X33_div_X34',
        'variance_30_35', 'energy_30_35'
    }

    # ==================================================
    # PHASE 1: LEAK-FREE FEATURE STABILITY FILTER CALCULATION
    # ==================================================
    logger.info("PHASE 1: Running fast CV pass to calculate feature stability metrics...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(df_train, df_train[TARGET_COL]))
    
    # Store importances across folds
    all_engineered_features = []
    fold_importances = []
    fold_top_features = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        df_train_fold = df_train.iloc[train_idx].copy()
        y_train = df_train_fold[TARGET_COL].values
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos = float(neg_count / max(1, pos_count))
        
        # Preprocessing & Feature Engineering
        preprocessor = RobustPreprocessor(scale_features=True, columns_with_nan=global_columns_with_nan)
        df_train_fold_proc = preprocessor.fit_transform(df_train_fold, feature_cols)
        proc_feature_cols = [c for c in df_train_fold_proc.columns if c not in [ID_COL, TARGET_COL]]
        
        engineer = LeakageFreeFeatureEngineer(seed=SEED)
        df_train_fold_eng = engineer.fit_transform(df_train_fold_proc, proc_feature_cols)
        final_feature_cols = [c for c in df_train_fold_eng.columns if c not in [ID_COL, TARGET_COL]]
        
        if fold_idx == 0:
            all_engineered_features = list(final_feature_cols)
            
        X_train_fold = df_train_fold_eng[final_feature_cols].values
        
        # Fit fast selector
        selector = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            random_state=SEED,
            tree_method='hist',
            n_jobs=-1
        )
        selector.fit(X_train_fold, y_train, verbose=False)
        
        importances = selector.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        
        fold_importances.append(importances)
        # Store sorted feature lists
        sorted_feats = [final_feature_cols[i] for i in sorted_idx]
        fold_top_features.append(sorted_feats)
        
    # Calculate Mean, Std, Stability, and Top15 counts
    stability_scores = {}
    feat_top15_counts = {feat: 0 for feat in all_engineered_features}
    
    for f_idx, sorted_feats in enumerate(fold_top_features):
        for feat in sorted_feats[:15]:
            feat_top15_counts[feat] += 1
            
    for f_idx, feat in enumerate(all_engineered_features):
        feat_vals = [fold_importances[i][f_idx] for i in range(5)]
        mean_imp = np.mean(feat_vals)
        std_imp = np.std(feat_vals)
        stability_scores[feat] = mean_imp / (std_imp + 1e-5)
        
    # ==================================================
    # PHASE 2: HYBRID STABILITY FILTER AND OVERLAP RELAXATION
    # ==================================================
    logger.info("PHASE 2: Evaluating hybrid feature stability and Jaccard overlap...")
    
    stability_thresh = 0.8
    top_n_to_force = 15
    
    while True:
        fold_selections = []
        for fold_idx in range(5):
            selected = []
            for feat in all_engineered_features:
                is_protected = feat in protected_set
                is_stable = (stability_scores.get(feat, 0.0) >= stability_thresh)
                is_top15 = (feat_top15_counts.get(feat, 0) >= 3)
                in_fold_top_n = feat in fold_top_features[fold_idx][:top_n_to_force]
                
                if is_protected or is_stable or is_top15 or in_fold_top_n:
                    selected.append(feat)
            fold_selections.append(set(selected))
            
        # Compute pairwise Jaccard index
        overlaps = []
        for i in range(5):
            for j in range(i+1, 5):
                u = fold_selections[i] | fold_selections[j]
                intersect = fold_selections[i] & fold_selections[j]
                jaccard = len(intersect) / len(u) if len(u) > 0 else 0.0
                overlaps.append(jaccard)
        avg_overlap = np.mean(overlaps)
        
        if avg_overlap >= 0.60 or top_n_to_force >= 50:
            break
            
        logger.warning(f"Feature selection unstable (Jaccard overlap = {avg_overlap:.2%}). Auto-relaxing (top_n={top_n_to_force}, thresh={stability_thresh:.2f})...")
        top_n_to_force += 5
        stability_thresh -= 0.15
        stability_thresh = max(0.1, stability_thresh)
        
    if avg_overlap < 0.60:
        logger.warning(f"Feature selection unstable (overlap = {avg_overlap:.2%}) even after relaxation.")
    else:
        logger.success(f"Feature selection stable! Jaccard fold overlap = {avg_overlap:.2%}")

    # ==================================================
    # PHASE 3: MAIN TRAINING RUN WITH SELECTION MASKS
    # ==================================================
    logger.info("PHASE 3: Running main Stratified CV with stability selection masks...")
    
    models_to_train = ['xgboost', 'catboost']
    oof_predictions = {model: np.zeros(len(df_train)) for model in models_to_train}
    y_true_all = df_train[TARGET_COL].values
    
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        df_train_fold = df_train.iloc[train_idx].copy()
        df_val_fold = df_train.iloc[val_idx].copy()
        
        y_train = df_train_fold[TARGET_COL].values
        y_val = df_val_fold[TARGET_COL].values
        
        # Dynamic scale_pos_weight
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos = float(neg_count / max(1, pos_count))
        
        # Preprocess
        preprocessor = RobustPreprocessor(scale_features=True, columns_with_nan=global_columns_with_nan)
        df_train_fold_proc = preprocessor.fit_transform(df_train_fold, feature_cols)
        df_val_fold_proc = preprocessor.transform(df_val_fold)
        save_model(preprocessor, os.path.join(MODELS_DIR, f"preprocessor_fold_{fold_idx}_rep_0.joblib"))
        
        proc_feature_cols = [c for c in df_train_fold_proc.columns if c not in [ID_COL, TARGET_COL]]
        
        # Feature Engineering
        engineer = LeakageFreeFeatureEngineer(seed=SEED)
        df_train_fold_eng = engineer.fit_transform(df_train_fold_proc, proc_feature_cols)
        df_val_fold_eng = engineer.transform(df_val_fold_proc)
        save_model(engineer, os.path.join(MODELS_DIR, f"feature_engineer_fold_{fold_idx}_rep_0.joblib"))
        
        # Apply selection mask for this fold
        selected_features = list(fold_selections[fold_idx])
        joblib.dump(selected_features, os.path.join(MODELS_DIR, f"selected_features_fold_{fold_idx}.joblib"))
        
        # Filter matrices
        X_train_sel = df_train_fold_eng[selected_features].values
        X_val_sel = df_val_fold_eng[selected_features].values
        
        # XGBoost
        fold_xgb_cfg = xgb_params_current.copy()
        fold_xgb_cfg['scale_pos_weight'] = scale_pos
        xgb_clf = XGBClassifier(**fold_xgb_cfg)
        xgb_clf.fit(
            X_train_sel, y_train,
            eval_set=[(X_val_sel, y_val)],
            verbose=False
        )
        oof_predictions['xgboost'][val_idx] = xgb_clf.predict_proba(X_val_sel)[:, 1]
        save_model(xgb_clf, os.path.join(MODELS_DIR, f"xgboost_fold_{fold_idx}_rep_0.joblib"))
        
        # CatBoost
        fold_cat_cfg = cat_params_current.copy()
        fold_cat_cfg['scale_pos_weight'] = scale_pos
        cat_clf = CatBoostClassifier(**fold_cat_cfg)
        cat_clf.fit(
            X_train_sel, y_train,
            eval_set=[(X_val_sel, y_val)]
        )
        oof_predictions['catboost'][val_idx] = cat_clf.predict_proba(X_val_sel)[:, 1]
        save_model(cat_clf, os.path.join(MODELS_DIR, f"catboost_fold_{fold_idx}_rep_0.joblib"))
        
    # Check OOF predictions correlation
    corr, _ = pearsonr(oof_predictions['xgboost'], oof_predictions['catboost'])
    logger.success(f"Out-of-Fold prediction Pearson Correlation: {corr:.4f}")
    
    # Save the selected hyperparameters for ensembling stage
    hyperparameter_meta = {
        'xgb_params': xgb_params_current,
        'cat_params': cat_params_current
    }
    joblib.dump(hyperparameter_meta, os.path.join(MODELS_DIR, 'hyperparameter_meta.joblib'))
    
    # Save OOF predictions and targets
    oof_data = {
        'oof_predictions': oof_predictions,
        'y_true': y_true_all
    }
    joblib.dump(oof_data, os.path.join(MODELS_DIR, 'oof_data.joblib'))
    logger.success("Saved out-of-fold predictions to models/oof_data.joblib")
    
    # ==================================================
    # PHASE 4: FIT FINAL MODELS ON COMPLETE DATASET
    # ==================================================
    logger.info("PHASE 4: Fitting final production-grade models on entire dataset...")
    global_scale_pos = float(np.sum(y_true_all == 0) / np.sum(y_true_all == 1))
    
    full_preprocessor = RobustPreprocessor(scale_features=True, columns_with_nan=global_columns_with_nan)
    df_train_proc = full_preprocessor.fit_transform(df_train, feature_cols)
    save_model(full_preprocessor, os.path.join(MODELS_DIR, "preprocessor_full.joblib"))
    save_model(full_preprocessor, os.path.join(MODELS_DIR, "preprocessor_full.pkl"))
    
    proc_feature_cols = [c for c in df_train_proc.columns if c not in [ID_COL, TARGET_COL]]
    
    full_engineer = LeakageFreeFeatureEngineer(seed=SEED)
    df_train_eng = full_engineer.fit_transform(df_train_proc, proc_feature_cols)
    save_model(full_engineer, os.path.join(MODELS_DIR, "feature_engineer_full.joblib"))
    save_model(full_engineer, os.path.join(MODELS_DIR, "feature_engineer_full.pkl"))
    
    # final features full: the union of all fold feature selections
    selected_features_full = list(set.union(*fold_selections))
    joblib.dump(selected_features_full, os.path.join(MODELS_DIR, 'final_features.joblib'))
    logger.success(f"Final full feature union completed: kept {len(selected_features_full)} features out of {len(all_engineered_features)}")
    
    X_full_sel = df_train_eng[selected_features_full].values
    
    # XGBoost
    xgb_params_full = xgb_params_current.copy()
    xgb_params_full['scale_pos_weight'] = global_scale_pos
    xgb_full = XGBClassifier(**xgb_params_full)
    xgb_full.fit(X_full_sel, y_true_all, verbose=False)
    save_model(xgb_full, os.path.join(MODELS_DIR, "xgboost.pkl"))
    save_model(xgb_full, os.path.join(MODELS_DIR, "xgboost.joblib"))
    logger.success("Fitted and saved xgboost.pkl")
    
    # CatBoost
    cat_params_full = cat_params_current.copy()
    cat_params_full['scale_pos_weight'] = global_scale_pos
    cat_full = CatBoostClassifier(**cat_params_full)
    cat_full.fit(X_full_sel, y_true_all, verbose=0)
    save_model(cat_full, os.path.join(MODELS_DIR, "catboost.pkl"))
    save_model(cat_full, os.path.join(MODELS_DIR, "catboost.joblib"))
    logger.success("Fitted and saved catboost.pkl")
    
    logger.success("All training phases complete!")

if __name__ == '__main__':
    run_training()
