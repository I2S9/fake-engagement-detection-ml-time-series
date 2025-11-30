"""
Training pipeline for baseline models.

This module provides functions to train baseline models on temporal features.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

from src.models.baselines import create_baseline_model, BaselineModel


def prepare_data(
    features_df: pd.DataFrame,
    label_column: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Prepare data for training by splitting features and labels.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with features and labels
    label_column : str
        Name of the label column
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, feature_names
    """
    # separate features and labels
    feature_columns = [col for col in features_df.columns if col not in [label_column, "id"]]
    X = features_df[feature_columns].values
    y = features_df[label_column].values

    # convert labels to binary (normal=0, fake=1)
    y_binary = np.where(y == "fake", 1, 0)

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
    )

    return X_train, X_test, y_train, y_test, feature_columns


def train_baseline_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_kwargs: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
) -> BaselineModel:
    """
    Train a baseline model.

    Parameters
    ----------
    model_type : str
        Type of model to train
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    model_kwargs : dict, optional
        Additional arguments for the model
    save_path : str, optional
        Path to save the trained model

    Returns
    -------
    BaselineModel
        Trained model
    """
    if model_kwargs is None:
        model_kwargs = {}

    # create and train model
    model = create_baseline_model(model_type, **model_kwargs)

    print(f"Training {model_type}...")
    model.fit(X_train, y_train)
    print(f"Training completed.")

    # save model if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"Model saved to {save_path}")

    return model


def train_multiple_baselines(
    features_df: pd.DataFrame,
    model_types: list,
    test_size: float = 0.2,
    random_state: int = 42,
    model_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, Tuple[BaselineModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Train multiple baseline models.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with features and labels
    model_types : list
        List of model types to train
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
    model_kwargs : dict, optional
        Dictionary mapping model_type to kwargs for each model
    save_dir : str, optional
        Directory to save trained models

    Returns
    -------
    dict
        Dictionary mapping model_type to (model, X_test, y_test, y_pred, y_proba)
    """
    # prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data(
        features_df, test_size=test_size, random_state=random_state
    )

    if model_kwargs is None:
        model_kwargs = {}

    results = {}

    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type}")
        print(f"{'='*60}")

        try:
            # get model-specific kwargs
            kwargs = model_kwargs.get(model_type, {})

            # train model
            save_path = None
            if save_dir is not None:
                save_path = os.path.join(save_dir, f"{model_type}.pkl")

            model = train_baseline_model(
                model_type, X_train, y_train, model_kwargs=kwargs, save_path=save_path
            )

            # make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            results[model_type] = (model, X_test, y_test, y_pred, y_proba)

            print(f"{model_type} training completed successfully.")

        except Exception as e:
            print(f"Error training {model_type}: {str(e)}")
            continue

    return results

