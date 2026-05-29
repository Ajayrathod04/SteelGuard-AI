import numpy as np
import pandas as pd

class LeakageFreeFeatureEngineer:
    """
    Kaggle-grade leakage-free stable feature engineer for SteelGuard-AI.
    Focuses on robust row-wise statistical aggregates, physical domain ratios/differences,
    row energy, and Shannon entropy without stateful data leakage.
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.feature_cols = []
        self.is_fitted = False

    def fit(self, df, feature_cols):
        """
        Fit stores feature columns. No stateful estimators are fitted.
        """
        self.feature_cols = list(feature_cols)
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
        transformed_df['row_skew'] = X.skew(axis=1)
        transformed_df['row_kurt'] = X.kurtosis(axis=1)
        
        # --- 2. High-Signal Energy and Entropy ---
        # row_energy: sum of squares
        transformed_df['row_energy'] = (X ** 2).sum(axis=1)
        
        # row_entropy: Shannon entropy of normalized row features
        row_min = X.min(axis=1)
        # Shift values to be positive for valid normalized probability distribution
        X_shifted = X.subtract(row_min, axis=0) + 1e-5
        row_sum = X_shifted.sum(axis=1)
        p = X_shifted.div(row_sum, axis=0)
        transformed_df['row_entropy'] = - (p * np.log(p + 1e-9)).sum(axis=1)
        
        # --- 3. Domain & Physical Sensor Interactions ---
        eps = 1e-6
        # X15 / X30
        if 'X15' in self.feature_cols and 'X30' in self.feature_cols:
            transformed_df['X15_div_X30'] = transformed_df['X15'] / (transformed_df['X30'] + eps)
            
        # X15 - X30
        if 'X15' in self.feature_cols and 'X30' in self.feature_cols:
            transformed_df['X15_minus_X30'] = transformed_df['X15'] - transformed_df['X30']
            
        # X31 * X32
        if 'X31' in self.feature_cols and 'X32' in self.feature_cols:
            transformed_df['X31_mul_X32'] = transformed_df['X31'] * transformed_df['X32']
            
        # X33 / X34
        if 'X33' in self.feature_cols and 'X34' in self.feature_cols:
            transformed_df['X33_div_X34'] = transformed_df['X33'] / (transformed_df['X34'] + eps)
            
        # variance_30_35: variance across columns X30 to X35
        cols_30_35 = [f'X{i}' for i in range(30, 36) if f'X{i}' in self.feature_cols]
        if cols_30_35:
            transformed_df['variance_30_35'] = X[cols_30_35].var(axis=1)
            # energy_30_35: sum of squares across columns X30 to X35
            transformed_df['energy_30_35'] = (X[cols_30_35] ** 2).sum(axis=1)
            
        return transformed_df

    def fit_transform(self, df, feature_cols):
        return self.fit(df, feature_cols).transform(df)
