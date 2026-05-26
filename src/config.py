import os
import sys

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# Ensure directories exist
for path in [MODELS_DIR, SUBMISSIONS_DIR, REPORTS_DIR, PLOTS_DIR]:
    os.makedirs(path, exist_ok=True)

# Data columns
TARGET_COL = 'Y'
ID_COL = 'CoilID'

# Add BASE_DIR and BASE_DIR/configs to sys.path
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'configs'))

from configs.config import RANDOM_STATE

# CV Parameters
SEED = RANDOM_STATE
N_SPLITS = 5
N_REPEATS = 2

XGB_PARAMS = {
    'random_state': SEED,
    'n_estimators': 400,
    'learning_rate': 0.03,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 5.0,
    'tree_method': 'hist',
    'eval_metric': 'logloss',
    'early_stopping_rounds': 50
}

LGB_PARAMS = {
    'random_state': SEED,
    'n_estimators': 400,
    'learning_rate': 0.03,
    'max_depth': 3,
    'num_leaves': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 5.0,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_jobs': -1,
    'verbose': -1
}

CAT_PARAMS = {
    'random_seed': SEED,
    'iterations': 400,
    'learning_rate': 0.03,
    'depth': 3,
    'l2_leaf_reg': 5.0,
    'eval_metric': 'Logloss',
    'verbose': 0,
    'early_stopping_rounds': 50
}

RF_PARAMS = {
    'random_state': SEED,
    'n_estimators': 200,
    'max_depth': 8,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'class_weight': 'balanced_subsample',
    'n_jobs': -1
}

ET_PARAMS = {
    'random_state': SEED,
    'n_estimators': 200,
    'max_depth': 8,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'class_weight': 'balanced_subsample',
    'n_jobs': -1
}
