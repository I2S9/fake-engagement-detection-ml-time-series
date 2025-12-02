"""
Interpretability module for Risk ML pipeline.
Provides feature importance, SHAP values, and explanations for model predictions.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not available. Install with: pip install shap")

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def compute_feature_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    method: str = "permutation"
) -> Dict[str, float]:
    """
    Compute feature importance for a model.
    
    Args:
        model: Trained model (sklearn-compatible)
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        method: Method to use ('permutation' or 'tree' for tree-based models)
    
    Returns:
        Dictionary mapping feature names to importance scores
    """
    if feature_names is None:
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    if method == "permutation":
        # Permutation importance
        perm_importance = permutation_importance(
            model, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )
        importances = perm_importance.importances_mean
    elif method == "tree" and hasattr(model, 'feature_importances_'):
        # Tree-based feature importance
        importances = model.feature_importances_
    else:
        # Fallback: use coefficients for linear models
        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
        else:
            raise ValueError(f"Method {method} not supported for this model type")
    
    # Normalize to sum to 1
    importances = importances / (importances.sum() + 1e-10)
    
    return dict(zip(feature_names, importances))


def compute_shap_values(
    model,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    n_samples: int = 100
) -> Tuple[np.ndarray, Optional[shap.Explainer]]:
    """
    Compute SHAP values for model interpretability.
    
    Args:
        model: Trained model
        X: Feature matrix
        feature_names: List of feature names
        n_samples: Number of samples to use for SHAP (for speed)
    
    Returns:
        Tuple of (SHAP values, SHAP explainer)
    """
    if not HAS_SHAP:
        warnings.warn("SHAP not available. Returning zeros.")
        return np.zeros((len(X), X.shape[1])), None
    
    if feature_names is None:
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Sample data for speed
    if len(X) > n_samples:
        X_sample = X.sample(n=n_samples, random_state=42)
    else:
        X_sample = X
    
    try:
        # Create explainer based on model type
        if hasattr(model, 'predict_proba'):
            # Tree-based or sklearn model
            explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict_proba, X_sample.iloc[:50] if hasattr(X_sample, 'iloc') else X_sample[:50])
        else:
            # Fallback to KernelExplainer
            explainer = shap.KernelExplainer(model.predict, X_sample.iloc[:50] if hasattr(X_sample, 'iloc') else X_sample[:50])
        
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        return shap_values, explainer
    except Exception as e:
        warnings.warn(f"SHAP computation failed: {e}. Returning zeros.")
        return np.zeros((len(X_sample), X_sample.shape[1])), None


def explain_prediction(
    model,
    X_instance: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    top_k: int = 10
) -> Dict[str, any]:
    """
    Explain a single prediction with feature contributions.
    
    Args:
        model: Trained model
        X_instance: Single instance to explain
        feature_names: List of feature names
        top_k: Number of top features to return
    
    Returns:
        Dictionary with prediction, probability, and top contributing features
    """
    if feature_names is None:
        if hasattr(X_instance, 'columns'):
            feature_names = X_instance.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X_instance.shape[1])]
    
    # Get prediction
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_instance)[0]
        prediction = model.predict(X_instance)[0]
    else:
        proba = [0.5, 0.5]  # Dummy
        prediction = model.predict(X_instance)[0]
    
    # Get feature importance
    try:
        importance_dict = compute_feature_importance(
            model, X_instance, np.array([prediction]), feature_names
        )
        top_features = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
    except:
        top_features = []
    
    return {
        "prediction": int(prediction),
        "probability": float(proba[1] if len(proba) > 1 else proba[0]),
        "top_features": dict(top_features),
        "explanation": f"Predicted as {'FAKE' if prediction == 1 else 'NORMAL'} "
                      f"with {proba[1] if len(proba) > 1 else proba[0]:.2%} confidence. "
                      f"Top contributing features: {', '.join([f[0] for f in top_features[:3]])}"
    }


def plot_feature_importance(
    importance_dict: Dict[str, float],
    top_k: int = 20,
    title: str = "Feature Importance"
):
    """
    Plot feature importance as a horizontal bar chart.
    
    Args:
        importance_dict: Dictionary mapping feature names to importance scores
        top_k: Number of top features to plot
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    # Sort and get top k
    sorted_features = sorted(
        importance_dict.items(), key=lambda x: x[1], reverse=True
    )[:top_k]
    
    features = [f[0] for f in sorted_features]
    importances = [f[1] for f in sorted_features]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
    
    y_pos = np.arange(len(features))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(features)))
    
    ax.barh(y_pos, importances, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel("Importance Score", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig, ax

