import os
import matplotlib
matplotlib.use('Agg')  # Headless mode to prevent GUI crashes
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
import shap

# Set styling
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.facecolor': '#1E1E1E',
    'axes.facecolor': '#1E1E1E',
    'text.color': '#FFFFFF',
    'axes.labelcolor': '#FFFFFF',
    'xtick.color': '#FFFFFF',
    'ytick.color': '#FFFFFF',
    'axes.edgecolor': '#444444'
})

def plot_target_distribution(df, target_col, save_path):
    """
    Plot target imbalance.
    """
    plt.figure(figsize=(6, 5))
    counts = df[target_col].value_counts()
    ax = sns.barplot(x=counts.index, y=counts.values, palette=["#3498db", "#e74c3c"])
    plt.title("Target Distribution Imbalance (Y)", fontsize=14, pad=15)
    plt.xlabel("Defect (Y)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    
    # Annotate percentages
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}\n({height/total*100:.2f}%)',
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center',
                    xytext=(0, 0), textcoords='offset points',
                    color='white', fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1E1E1E', dpi=300)
    plt.close()

def plot_missing_values(df, save_path):
    """
    Plot columns with missing values.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        # Create an empty plot indicating no missing values
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "No Missing Values Found!", ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.savefig(save_path, facecolor='#1E1E1E', dpi=300)
        plt.close()
        return

    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing.values, y=missing.index, palette="viridis")
    plt.title("Missing Values Count per Feature", fontsize=14, pad=15)
    plt.xlabel("Number of Missing Values", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1E1E1E', dpi=300)
    plt.close()

def plot_correlation_heatmap(df, feature_cols, save_path):
    """
    Plot correlation heatmap for feature columns.
    """
    plt.figure(figsize=(16, 14))
    corr = df[feature_cols].corr()
    
    # We can plot only a subset or mask upper triangle for readability
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8},
                xticklabels=True, yticklabels=True)
                
    plt.title("Feature Correlation Matrix", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1E1E1E', dpi=300)
    plt.close()

def plot_feature_distributions(df, features_to_plot, save_path):
    """
    Plot histograms of a subset of features.
    """
    n_features = len(features_to_plot)
    fig, axes = plt.subplots(int(np.ceil(n_features / 2)), 2, figsize=(12, 4 * np.ceil(n_features / 2)))
    axes = axes.flatten()
    
    for i, col in enumerate(features_to_plot):
        sns.histplot(data=df, x=col, hue="Y", kde=True, ax=axes[i], palette=["#3498db", "#e74c3c"], bins=30)
        axes[i].set_title(f"Distribution of {col}", fontsize=12)
        axes[i].set_xlabel("")
        
    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1E1E1E', dpi=300)
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, opt_threshold, save_path):
    """
    Plot Precision-Recall Curve with the selected threshold marked.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#1abc9c', lw=3, label='Precision-Recall Curve')
    
    # Find recall and precision at the optimal threshold
    # Note: thresholds has length len(precision)-1
    idx = np.argmin(np.abs(thresholds - opt_threshold))
    opt_prec = precision[idx]
    opt_rec = recall[idx]
    
    plt.plot(opt_rec, opt_prec, 'ro', markersize=10, label=f'Optimized Threshold = {opt_threshold:.3f}\n(Recall={opt_rec:.3f}, Precision={opt_prec:.3f})')
    
    plt.axhline(y=0.90, color='r', linestyle='--', alpha=0.7, label='Precision constraint (>90%)')
    
    plt.title("Precision-Recall Curve (Out-of-Fold Blended Predictions)", fontsize=14, pad=15)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.legend(loc="lower left", framealpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1E1E1E', dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Plot beautiful confusion matrix.
    """
    from sklearn.metrics import confusion_matrix as sk_cm
    cm = sk_cm(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    # Custom steel-blue mapping
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                annot_kws={"size": 14, "weight": "bold"},
                xticklabels=["Normal (0)", "Defect (1)"],
                yticklabels=["Normal (0)", "Defect (1)"])
                
    plt.title("Confusion Matrix (Out-of-Fold Optimized Predictions)", fontsize=14, pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1E1E1E', dpi=300)
    plt.close()

def plot_shap_summary(model, X_train, save_path):
    """
    Plot SHAP Summary using tree explainer.
    Works for Tree models like LightGBM or XGBoost.
    """
    plt.figure(figsize=(10, 6))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # For multiclass or binary class, SHAP values may return list or array.
    # In shap >= 0.40, binary shap values is a list/array with shape (n_samples, n_features).
    # Handle both cases.
    if isinstance(shap_values, list) and len(shap_values) == 2:
        # Binary target returns list of [shap_class_0, shap_class_1]
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values
        
    shap.summary_plot(shap_values_class1, X_train, show=False, max_display=15)
    plt.title("SHAP Feature Importance (Top 15)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1E1E1E', dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names, save_path, max_features=15):
    """
    Plot feature importances for a tree-based model.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):
        importances = model.get_feature_importance()
    else:
        # Fallback if no importances exist
        return
        
    import numpy as np
    indices = np.argsort(importances)[::-1]
    
    # Take top max_features
    top_indices = indices[:max_features]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_names, palette="viridis")
    plt.title("Feature Importance Analysis (Top 15 Process Parameters)", fontsize=14, pad=15)
    plt.xlabel("Relative Importance Score", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1E1E1E', dpi=300)
    plt.close()

def plot_precision_recall_vs_threshold(y_true, y_prob, save_path, opt_threshold=None):
    """
    Plot Precision and Recall curves across different thresholds.
    """
    from sklearn.metrics import precision_score, recall_score
    thresholds = np.linspace(0.01, 0.99, 100)
    precisions = []
    recalls = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(float)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label="Precision", color="#3498db", lw=2.5)
    plt.plot(thresholds, recalls, label="Recall", color="#e74c3c", lw=2.5)
    
    if opt_threshold is not None:
        plt.axvline(x=opt_threshold, color="#1abc9c", linestyle="--", lw=2, 
                    label=f"Optimized Threshold t* = {opt_threshold:.4f}")
        # Add indicator for the >= 90% Precision requirement
        plt.axhline(y=0.90, color="#f1c40f", linestyle=":", lw=1.5, label="Precision target constraint (90%)")
        
    plt.title("Threshold Selection & Optimization Frontier", fontsize=14, pad=15)
    plt.xlabel("Probability Decision Threshold", fontsize=12)
    plt.ylabel("Score Value", fontsize=12)
    plt.legend(loc="lower left", framealpha=0.2)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1E1E1E', dpi=300)
    plt.close()
