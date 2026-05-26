import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

class LeakageFreeFeatureEngineer:
    """
    Kaggle-grade leakage-free feature engineer.
    Fits all stateful components (PCA, KMeans, IsolationForest) ONLY on the training fold
    and applies transform to validation and test folds.
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.pca = PCA(n_components=5, random_state=seed)
        self.kmeans = KMeans(n_clusters=4, random_state=seed, n_init=10)
        self.iso_forest = IsolationForest(contamination=0.05, random_state=seed, n_jobs=-1)
        self.feature_cols = []
        self.medians = {}
        self.stds = {}
        self.is_fitted = False
        
        # Correlated feature groups identified in EDA
        self.block_a = ['X29', 'X30', 'X31', 'X32', 'X33']
        self.block_b = ['X10', 'X13', 'X15', 'X16']

    def fit(self, df, feature_cols):
        """
        Fit stateful models on training data.
        """
        self.feature_cols = list(feature_cols)
        
        # Prepare subset of data for fitting (after imputation, which happens before FE)
        X = df[self.feature_cols].copy()
        
        # Fit Isolation Forest for anomaly score
        self.iso_forest.fit(X)
        
        # Store medians and stds for drift and outlier features
        self.medians = X.median().to_dict()
        self.stds = X.std().to_dict()
        
        self.is_fitted = True
        return self

    def transform(self, df):
        """
        Transform dataset and engineer features.
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transforming.")
            
        transformed_df = df.copy()
        X = transformed_df[self.feature_cols].copy()
        
        # --- 1. Global Row-wise Statistics ---
        transformed_df['row_mean'] = X.mean(axis=1)
        transformed_df['row_std'] = X.std(axis=1)
        transformed_df['row_min'] = X.min(axis=1)
        transformed_df['row_max'] = X.max(axis=1)
        transformed_df['row_range'] = transformed_df['row_max'] - transformed_df['row_min']
        transformed_df['row_skew'] = X.skew(axis=1)
        transformed_df['row_kurt'] = X.kurtosis(axis=1)
        
        # --- 1B. Sensor Drift and Outlier Features ---
        top_indicators = ['X31', 'X13', 'X35', 'X30', 'X10']
        for col in top_indicators:
            if col in self.feature_cols:
                med = self.medians.get(col, 0.0)
                std_val = self.stds.get(col, 1.0)
                transformed_df[f'z_score_{col}'] = (X[col] - med) / (std_val + 1e-6)
                transformed_df[f'drift_{col}'] = X[col] - med
        
        # --- 2. Block-specific Statistical Features ---
        transformed_df['block_a_mean'] = X[self.block_a].mean(axis=1)
        transformed_df['block_a_std'] = X[self.block_a].std(axis=1)
        transformed_df['block_b_mean'] = X[self.block_b].mean(axis=1)
        transformed_df['block_b_std'] = X[self.block_b].std(axis=1)

        # --- 3. Collinear Pair Differences and Ratios ---
        eps = 1e-6
        # Pair 1: X31 and X30
        transformed_df['X31_minus_X30'] = transformed_df['X31'] - transformed_df['X30']
        transformed_df['X31_div_X30'] = transformed_df['X31'] / (transformed_df['X30'] + eps)
        
        # Pair 2: X13 and X10
        transformed_df['X13_minus_X10'] = transformed_df['X13'] - transformed_df['X10']
        transformed_df['X13_div_X10'] = transformed_df['X13'] / (transformed_df['X10'] + eps)
        
        # Pair 3: X32 and X31
        transformed_df['X32_minus_X31'] = transformed_df['X32'] - transformed_df['X31']
        transformed_df['X32_div_X31'] = transformed_df['X32'] / (transformed_df['X31'] + eps)
        
        # Pair 4: X9 and X8
        transformed_df['X9_minus_X8'] = transformed_df['X9'] - transformed_df['X8']
        transformed_df['X9_div_X8'] = transformed_df['X9'] / (transformed_df['X8'] + eps)
        
        # Pair 5: X7 and X4
        transformed_df['X7_minus_X4'] = transformed_df['X7'] - transformed_df['X4']
        transformed_df['X7_div_X4'] = transformed_df['X7'] / (transformed_df['X4'] + eps)
        
        # --- 4. Anomaly Score (Isolation Forest) ---
        # decision_function returns anomaly scores (lower = more anomalous)
        transformed_df['anomaly_score'] = self.iso_forest.decision_function(X)
        
        return transformed_df

    def fit_transform(self, df, feature_cols):
        return self.fit(df, feature_cols).transform(df)
