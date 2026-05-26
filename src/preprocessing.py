import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

class RobustPreprocessor:
    """
    Leakage-free Preprocessor for SteelGuard-AI.
    Fits only on training folds and applies to validation and test folds.
    """
    def __init__(self, scale_features=True, columns_with_nan=None):
        self.scale_features = scale_features
        self.medians = {}
        self.scaler = RobustScaler()
        self.columns_with_nan = list(columns_with_nan) if columns_with_nan is not None else []
        self.feature_cols = []
        self.engineered_feature_cols = []

    def fit(self, df, feature_cols):
        """
        Fit imputation and scaling on training data.
        """
        self.feature_cols = list(feature_cols)
        
        # Identify columns with missing values in training if not pre-specified
        if not self.columns_with_nan:
            nan_counts = df[self.feature_cols].isnull().sum()
            self.columns_with_nan = list(nan_counts[nan_counts > 0].index)
        
        # Calculate medians for imputation
        for col in self.feature_cols:
            self.medians[col] = df[col].median()
            
        # Fit scaler on imputed training features
        imputed_df = df[self.feature_cols].copy()
        for col in self.feature_cols:
            imputed_df[col] = imputed_df[col].fillna(self.medians[col])
            
        if self.scale_features:
            self.scaler.fit(imputed_df[self.feature_cols])
            
        return self

    def transform(self, df):
        """
        Apply imputation, missingness indicator creation, and scaling.
        """
        transformed_df = df.copy()
        
        # 1. Create missingness indicators (before imputation)
        for col in self.columns_with_nan:
            transformed_df[f"{col}_isnull"] = transformed_df[col].isnull().astype(float)
            
        # 2. Impute missing values with training medians
        for col in self.feature_cols:
            transformed_df[col] = transformed_df[col].fillna(self.medians[col])
            
        # 3. Apply Robust Scaling
        if self.scale_features:
            scaled_data = self.scaler.transform(transformed_df[self.feature_cols])
            # Replace original features with scaled ones
            transformed_df[self.feature_cols] = scaled_data
            
        return transformed_df

    def fit_transform(self, df, feature_cols):
        return self.fit(df, feature_cols).transform(df)
