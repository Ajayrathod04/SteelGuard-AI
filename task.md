# Task Checklist - SteelGuard-AI Ultra Safe Pipeline

- [x] Reconfigure `src/preprocessing.py` to support baseline Robust Scaler imputations
- [x] Retain high-signal features in `src/feature_engineering.py`
- [x] Configure `src/train.py` for 5-Fold StratifiedKFold, leak-free Hybrid Feature Stability Filter, selection overlap auto-relaxation safety, and micro-hyperparameter tuning
- [x] Update `src/ensemble.py` to sweep thresholds narrowly in `[0.045, 0.07]`, optimizing for fold-penalized F2 score
- [x] Update `src/predict.py` to bag predictions across folds, apply hybrid selected feature masks, and soft monitor prediction counts
- [x] Run the complete optimized pipeline using `python src/run_all.py`
- [x] Run `python src/verify_submission.py` to validate formatting and prediction distribution
