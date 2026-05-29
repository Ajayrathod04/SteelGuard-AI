# Implementation Plan - SteelGuard-AI Ultra Safe Score Boost Mode (Hybrid Stability Hardened)

This plan details the architecture for configuring the SteelGuard-AI defect detection pipeline in **Ultra Safe Score Boost Mode** with your robust Hybrid Feature Stability Filter and automatic selection overlap safety limits.

---

## User Review Required

> [!IMPORTANT]
> - **Hybrid Feature Stability Filter**: Inside each cross-validation fold, we will perform feature pruning by retaining a feature if it satisfies **ANY** of the following:
>   1. **Stability Score** ($\text{Mean} / \text{Std} \ge 0.8$) across folds.
>   2. **Top-15 Importance**: Appears in the top-15 features by importance in at least 3 splits.
>   3. **Protected Domain Features**: Features belonging to the protected set (`X15`, `X30`, `X31`, `X32`, `X33`, `X34`, `X35`) and their engineered derivatives (`X15_div_X30`, `X15_minus_X30`, `X31_mul_X32`, `X33_div_X34`, `variance_30_35`, `energy_30_35`) will **NEVER** be pruned.
> - **Feature Selection Overlap & Auto-Relaxation**: We will calculate the pairwise Jaccard overlap between the features selected across the 5 splits. If the average overlap is **< 60%**, the script will print a warning:
>   `WARNING: Feature selection unstable (overlap = X.XX%)`
>   and **automatically relax** the pruning constraints (including more top features) until feature alignment is stable and rare signals are protected.
> - **Differentiated Micro-Tuning**: Regularizing settings:
>   - **XGBoost**: `max_depth=3`, `subsample=0.75`, `colsample_bytree=0.75`, `min_child_weight=2`, `gamma=0.1`, `reg_alpha=1.5`, `reg_lambda=6.0`.
>   - **CatBoost**: `depth=4`, `learning_rate=0.025`, `l2_leaf_reg=6.0`, `random_strength=0.2`, `bagging_temperature=0.2`.
> - **Narrow Threshold Search**: Sweep decision thresholds strictly in `[0.045, 0.07]`, optimizing for stability-penalized OOF F2-score.

---

## Proposed Changes

### 1. Preprocessing and Feature Scaling
#### [MODIFY] [src/preprocessing.py](file:///D:/program%20files/SteelGuard-AI/SteelGuard-AI/src/preprocessing.py)
- Retain the baseline Robust Preprocessor using `RobustScaler` and medians imputation.

### 2. Feature Engineering
#### [MODIFY] [src/feature_engineering.py](file:///D:/program%20files/SteelGuard-AI/SteelGuard-AI/src/feature_engineering.py)
- Retain row statistical aggregates, energy, Shannon entropy, and localized interactions.

### 3. Hybrid Stability Filter & Training Pipeline
#### [MODIFY] [src/train.py](file:///D:/program%20files/SteelGuard-AI/SteelGuard-AI/src/train.py)
- In the training loop:
  - First-pass: Fit a fast selector model across the 5 splits, collect feature importances.
  - Apply the **Hybrid Feature Stability Filter** (incorporating protected domain columns, stability score $\ge 0.8$, and top-15 occurrences in $\ge 3$ splits).
  - Measure pairwise selection overlap (Jaccard index). If average overlap $< 60\%$, automatically relax the constraints and expand fold masks.
  - Train our main classifiers **CatBoost** and **XGBoost** with our regularized micro-tuned hyperparameters.
- Track fold consistency, recall std, and OOF Pearson correlation.

### 4. Simplex Ensemble and Narrow Penalized Threshold Search
#### [MODIFY] [src/ensemble.py](file:///D:/program%20files/SteelGuard-AI/SteelGuard-AI/src/ensemble.py)
- Soft voting blend: `w_cat * cat_prob + w_xgb * xgb_prob`.
- Sweep thresholds strictly from `0.045` to `0.07`.
- Optimize the penalized score: $\text{OOF F2} - 0.5 \times \text{Std}(\text{Fold Recalls}) - 0.5 \times \text{Std}(\text{Fold Precisions})$.
- Print detailed metrics: Recall, Precision, PR-AUC, F2, fold variance, and correlation.

### 5. Synchronized Test Prediction
#### [MODIFY] [src/predict.py](file:///D:/program%20files/SteelGuard-AI/SteelGuard-AI/src/predict.py)
- Bag predictions across all 5 folds on the test set, applying the stability-filtered feature masks.
- Apply ensembling weights and penalized threshold.
- Log a soft monitoring check against the healthy **35–70** positive prediction range.
- Generate synchronized submissions in `submissions/expected_submission.csv` on both drives.

---

## Verification Plan

### Automated Verification
1. Run the entire pipeline using `python src/run_all.py`.
2. Confirm the pipeline completes training, feature stability filtering, and inference cleanly.
3. Validate the submission file using `src/verify_submission.py` to check format correctness.
