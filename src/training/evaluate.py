"""
Evaluation functions for baseline models.

This module provides functions to evaluate model performance with metrics
and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import seaborn as sns


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray, optional
        Predicted probabilities (for positive class)

    Returns
    -------
    dict
        Dictionary with metrics
    """
    metrics = {}

    # basic classification metrics
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # AUC if probabilities available
    if y_proba is not None:
        if y_proba.ndim > 1:
            # get probability for positive class (fake)
            y_proba_positive = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten()
        else:
            y_proba_positive = y_proba

        try:
            metrics["auc"] = roc_auc_score(y_true, y_proba_positive)
        except ValueError:
            metrics["auc"] = 0.0
    else:
        metrics["auc"] = 0.0

    # confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        metrics["true_negatives"] = 0
        metrics["false_positives"] = 0
        metrics["false_negatives"] = 0
        metrics["true_positives"] = 0
        metrics["false_positive_rate"] = 0.0

    return metrics


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    model_name : str
        Name of the model for legend
    ax : plt.Axes, optional
        Matplotlib axes to plot on

    Returns
    -------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if y_proba.ndim > 1:
        y_proba_positive = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten()
    else:
        y_proba_positive = y_proba

    fpr, tpr, _ = roc_curve(y_true, y_proba_positive)
    auc_score = roc_auc_score(y_true, y_proba_positive)

    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot Precision-Recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    model_name : str
        Name of the model for legend
    ax : plt.Axes, optional
        Matplotlib axes to plot on

    Returns
    -------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if y_proba.ndim > 1:
        y_proba_positive = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten()
    else:
        y_proba_positive = y_proba

    precision, recall, _ = precision_recall_curve(y_true, y_proba_positive)
    avg_precision = np.trapz(precision, recall)

    ax.plot(recall, precision, label=f"{model_name} (AP = {avg_precision:.3f})", linewidth=2)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_score_distributions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot distribution of prediction scores for normal vs fake.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    model_name : str
        Name of the model for title
    ax : plt.Axes, optional
        Matplotlib axes to plot on

    Returns
    -------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if y_proba.ndim > 1:
        y_proba_positive = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten()
    else:
        y_proba_positive = y_proba

    normal_scores = y_proba_positive[y_true == 0]
    fake_scores = y_proba_positive[y_true == 1]

    ax.hist(normal_scores, bins=30, alpha=0.6, label="Normal", color="blue", density=True)
    ax.hist(fake_scores, bins=30, alpha=0.6, label="Fake", color="red", density=True)
    ax.set_xlabel("Prediction Score (Fake Probability)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Score Distribution - {model_name}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    model_name : str
        Name of the model for title
    ax : plt.Axes, optional
        Matplotlib axes to plot on

    Returns
    -------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Normal", "Fake"],
        yticklabels=["Normal", "Fake"],
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")

    return ax


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    plot: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a model and optionally create visualizations.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray
        Predicted probabilities
    model_name : str
        Name of the model
    plot : bool
        Whether to create plots

    Returns
    -------
    dict
        Dictionary with metrics
    """
    # compute metrics
    metrics = compute_metrics(y_true, y_pred, y_proba)

    # print metrics
    print(f"\n{'='*60}")
    print(f"Evaluation Results - {model_name}")
    print(f"{'='*60}")
    print(f"AUC:        {metrics['auc']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-Score:   {metrics['f1']:.4f}")
    print(f"FPR:         {metrics['false_positive_rate']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  True Positives:  {metrics['true_positives']}")

    # create plots
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # ROC curve
        plot_roc_curve(y_true, y_proba, model_name=model_name, ax=axes[0, 0])

        # Precision-Recall curve
        plot_precision_recall_curve(y_true, y_proba, model_name=model_name, ax=axes[0, 1])

        # Score distributions
        plot_score_distributions(y_true, y_proba, model_name=model_name, ax=axes[1, 0])

        # Confusion matrix
        plot_confusion_matrix(y_true, y_pred, model_name=model_name, ax=axes[1, 1])

        plt.tight_layout()
        plt.show()

    return metrics


def compare_models(
    results: Dict[str, Tuple],
    plot: bool = True,
) -> pd.DataFrame:
    """
    Compare multiple models and create summary.

    Parameters
    ----------
    results : dict
        Dictionary mapping model_name to (model, X_test, y_test, y_pred, y_proba)
    plot : bool
        Whether to create comparison plots

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for all models
    """
    all_metrics = []

    for model_name, (model, X_test, y_test, y_pred, y_proba) in results.items():
        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["model"] = model_name
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.set_index("model")

    # print summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(metrics_df[["auc", "precision", "recall", "f1", "false_positive_rate"]])

    # create comparison plots
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # ROC curves comparison
        ax = axes[0, 0]
        for model_name, (model, X_test, y_test, y_pred, y_proba) in results.items():
            plot_roc_curve(y_test, y_proba, model_name=model_name, ax=ax)
        ax.set_title("ROC Curves Comparison", fontsize=14, fontweight="bold")

        # Precision-Recall curves comparison
        ax = axes[0, 1]
        for model_name, (model, X_test, y_test, y_pred, y_proba) in results.items():
            plot_precision_recall_curve(y_test, y_proba, model_name=model_name, ax=ax)
        ax.set_title("Precision-Recall Curves Comparison", fontsize=14, fontweight="bold")

        # Metrics bar chart
        ax = axes[1, 0]
        metrics_to_plot = ["auc", "precision", "recall", "f1"]
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(results)
        for idx, model_name in enumerate(results.keys()):
            values = [metrics_df.loc[model_name, m] for m in metrics_to_plot]
            ax.bar(x + idx * width, values, width, label=model_name, alpha=0.8)
        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Metrics Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (len(results) - 1) / 2)
        ax.set_xticklabels(metrics_to_plot)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Score distributions comparison
        ax = axes[1, 1]
        for model_name, (model, X_test, y_test, y_pred, y_proba) in results.items():
            if y_proba.ndim > 1:
                y_proba_positive = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten()
            else:
                y_proba_positive = y_proba
            ax.hist(
                y_proba_positive[y_test == 1],
                bins=20,
                alpha=0.5,
                label=f"{model_name} (Fake)",
                density=True,
            )
        ax.set_xlabel("Prediction Score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Fake Class Score Distributions", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return metrics_df

