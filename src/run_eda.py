import os
import pandas as pd

from config import DATA_DIR, PLOTS_DIR, TARGET_COL, ID_COL
from utils import SimpleLogger
from visualization import (
    plot_target_distribution, plot_missing_values,
    plot_correlation_heatmap, plot_feature_distributions
)

def run_eda():
    logger = SimpleLogger("SteelGuard-AI-EDA")
    logger.info("Starting Full Exploratory Data Analysis (EDA)...")
    
    # 1. Load Data
    train_path = os.path.join(DATA_DIR, 'train.csv')
    df_train = pd.read_csv(train_path)
    logger.success(f"Successfully loaded training data: {df_train.shape}")
    
    # Define features
    feature_cols = [col for col in df_train.columns if col not in [ID_COL, TARGET_COL]]
    
    # 2. Target Imbalance Analysis
    logger.info("Generating target imbalance analysis...")
    plot_target_distribution(df_train, TARGET_COL, os.path.join(PLOTS_DIR, 'target_distribution.png'))
    logger.success("Saved plots/target_distribution.png")
    
    # 3. Missing Value Analysis
    logger.info("Generating missing value analysis...")
    plot_missing_values(df_train, os.path.join(PLOTS_DIR, 'missing_values.png'))
    logger.success("Saved plots/missing_values.png")
    
    # 4. Feature Distribution Analysis (Top Correlated Features)
    logger.info("Generating continuous feature distribution analysis for top indicators...")
    # X35, X13, X34, X36 are highly correlated with target Y; X31 represents a key sensor block feature
    top_indicators = ['X35', 'X13', 'X34', 'X36', 'X31']
    plot_feature_distributions(df_train, top_indicators, os.path.join(PLOTS_DIR, 'feature_distributions.png'))
    logger.success("Saved plots/feature_distributions.png")
    
    # 5. Correlation Heatmap
    logger.info("Generating correlation heatmap for original X1-X49 features...")
    plot_correlation_heatmap(df_train, feature_cols, os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))
    logger.success("Saved plots/correlation_heatmap.png")
    
    logger.success("Full EDA complete! All exploratory visualizations saved to the 'plots/' directory.")

if __name__ == '__main__':
    run_eda()
