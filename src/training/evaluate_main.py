"""
Main evaluation script for fake engagement detection models.

This script evaluates trained models and generates:
- ROC curves
- Score distributions
- Metrics (AUC, Precision, Recall, F1, FPR)
- Saves outputs to outputs/ directory

Usage:
    python src/training/evaluate_main.py --config config/config.yaml
    python -m src.training.evaluate_main --config config/config.yaml
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import os
import json
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend
import matplotlib.pyplot as plt

# add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocess import load_and_preprocess
from src.data.sequence_preparation import prepare_sequences_for_training
from src.data.dataset import create_dataloaders_from_dict
from src.features.temporal_features import extract_temporal_features
from src.models.baselines import create_baseline_model
from src.training.train import train_multiple_baselines
from src.training.evaluate import (
    evaluate_sequential_model,
    compute_metrics,
    plot_roc_curve,
    plot_score_distributions,
)
from src.utils.config import load_config, update_config_with_data


def load_baseline_model(model_path: str, model_type: str):
    """Load a trained baseline model."""
    model = create_baseline_model(model_type)
    model.load(model_path)
    return model


def load_sequential_model(model_path: str, model_type: str, config: Dict[str, Any], device: torch.device):
    """Load a trained sequential model."""
    from src.models.lstm import create_lstm_model
    from src.models.tcn import create_tcn_model
    from src.models.autoencoder import create_autoencoder_model

    model_config = config.get(model_type, {})
    input_size = model_config.get("input_size")
    seq_len = config.get("data", {}).get("seq_len", 48)

    if input_size is None:
        raise ValueError(f"input_size must be specified in config for {model_type}")

    if model_type == "lstm":
        model = create_lstm_model(model_config)
    elif model_type == "tcn":
        model = create_tcn_model(model_config)
    elif model_type == "autoencoder":
        model_config["seq_len"] = seq_len
        model = create_autoencoder_model(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def evaluate_baseline_model(model, features_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Evaluate a baseline model."""
    from src.training.train import prepare_data

    X_train, X_test, y_train, y_test, feature_names = prepare_data(
        features_df, test_size=test_size, random_state=random_state
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    return y_test, y_pred, y_proba


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate fake engagement detection models")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save evaluation outputs",
    )
    parser.add_argument(
        "--model-types",
        type=str,
        nargs="+",
        default=None,
        help="Model types to evaluate (e.g., lstm tcn autoencoder logistic_regression)",
    )

    args = parser.parse_args()

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # load configuration
    print("=" * 60)
    print("Loading configuration...")
    print("=" * 60)
    config = load_config(args.config)
    print(f"Configuration loaded from: {args.config}")

    # setup device
    device_config = config.get("training", {}).get("device", "cpu")
    if device_config == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(device_config if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # determine data path
    data_path = config.get("data", {}).get("data_path", "data/raw/engagement_timeseries.parquet")
    if not os.path.isabs(data_path):
        data_path = project_root / data_path

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # load and preprocess data
    print("\n" + "=" * 60)
    print("Loading and preprocessing data...")
    print("=" * 60)
    df = load_and_preprocess(
        file_path=str(data_path),
        target_timezone="UTC",
        resample_frequency="h",
        handle_missing=True,
        missing_method="forward",
        normalize=False,
    )
    print(f"Data shape: {df.shape}")
    print(f"Number of videos: {df['id'].nunique()}")

    all_metrics = {}
    all_results = {}

    # evaluate baseline models
    baseline_models = config.get("training", {}).get("baseline_models", [
        "logistic_regression",
        "random_forest",
        "isolation_forest",
    ])

    if args.model_types:
        baseline_models = [m for m in baseline_models if m in args.model_types]

    baseline_save_dir = config.get("training", {}).get("baseline_model_save_dir", "models/baselines")

    if baseline_models:
        print("\n" + "=" * 60)
        print("Evaluating baseline models...")
        print("=" * 60)

        # extract temporal features
        print("Extracting temporal features...")
        features_df = extract_temporal_features(
            df,
            id_column="id",
            timestamp_column="timestamp",
            aggregate_per_id=True,
        )

        # ensure label is present
        if "label" not in features_df.columns and "label" in df.columns:
            labels_per_id = df.groupby("id")["label"].first().reset_index()
            if "id" not in features_df.columns:
                if features_df.index.name == "id" or isinstance(features_df.index, pd.Index):
                    features_df = features_df.reset_index()
            if "id" in features_df.columns:
                features_df = features_df.merge(labels_per_id, on="id", how="left")

        if "label" not in features_df.columns:
            raise ValueError("Label column not found in features_df. Cannot evaluate baseline models.")

        for model_type in baseline_models:
            model_path = os.path.join(baseline_save_dir, f"{model_type}.pkl")
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found: {model_path}, skipping")
                continue

            print(f"\nEvaluating {model_type}...")
            try:
                model = load_baseline_model(model_path, model_type)
                y_test, y_pred, y_proba = evaluate_baseline_model(
                    model,
                    features_df,
                    test_size=config.get("data", {}).get("test_size", 0.2),
                    random_state=config.get("training", {}).get("random_seed", 42),
                )

                metrics = compute_metrics(y_test, y_pred, y_proba)
                all_metrics[model_type] = metrics
                all_results[model_type] = (y_test, y_pred, y_proba)

                print(f"  AUC: {metrics['auc']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1: {metrics['f1']:.4f}")

            except Exception as e:
                print(f"Error evaluating {model_type}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    # evaluate sequential models
    sequential_models = config.get("training", {}).get("sequential_models", ["lstm", "tcn", "autoencoder"])

    if args.model_types:
        sequential_models = [m for m in sequential_models if m in args.model_types]

    sequential_save_dir = config.get("training", {}).get("model_save_dir", "models/sequential")

    if sequential_models:
        print("\n" + "=" * 60)
        print("Evaluating sequential models...")
        print("=" * 60)

        # prepare sequences
        print("Preparing sequences...")
        seq_len = config.get("data", {}).get("seq_len", 48)
        stride = config.get("data", {}).get("stride", 1)

        sequences_dict = prepare_sequences_for_training(
            df,
            id_column="id",
            timestamp_column="timestamp",
            label_column="label",
            seq_len=seq_len,
            stride=stride,
            normalize=config.get("data", {}).get("normalize", True),
            normalization_method=config.get("data", {}).get("normalization_method", "standardize"),
            normalize_per_series=config.get("data", {}).get("normalize_per_series", False),
            test_size=config.get("data", {}).get("test_size", 0.2),
            val_size=config.get("data", {}).get("val_size", 0.1),
            random_state=config.get("training", {}).get("random_seed", 42),
        )

        # get input size from sequences
        X_train = sequences_dict["X_train"]
        input_size = X_train.shape[2]
        print(f"Input size: {input_size}")

        # update config with input size
        config = update_config_with_data(config, input_size, seq_len)

        # create dataloaders
        print("Creating DataLoaders...")
        batch_size = config.get("data", {}).get("batch_size", 32)
        num_workers = config.get("data", {}).get("num_workers", 0)
        pin_memory = config.get("data", {}).get("pin_memory", False)

        dataloaders = create_dataloaders_from_dict(
            sequences_dict,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        for model_type in sequential_models:
            model_path = os.path.join(sequential_save_dir, f"{model_type}_best.pth")
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found: {model_path}, skipping")
                continue

            print(f"\nEvaluating {model_type}...")
            try:
                model = load_sequential_model(model_path, model_type, config, device)
                y_test, y_pred, y_proba = evaluate_sequential_model(
                    model=model,
                    dataloader=dataloaders["test"],
                    device=device,
                    model_type=model_type,
                )

                metrics = compute_metrics(y_test, y_pred, y_proba)
                all_metrics[model_type] = metrics
                all_results[model_type] = (y_test, y_pred, y_proba)

                print(f"  AUC: {metrics['auc']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1: {metrics['f1']:.4f}")

            except Exception as e:
                print(f"Error evaluating {model_type}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    # generate visualizations
    if all_results:
        print("\n" + "=" * 60)
        print("Generating visualizations...")
        print("=" * 60)

        # ROC curve
        fig, ax = plt.subplots(figsize=(12, 10))
        for model_name, (y_test, y_pred, y_proba) in all_results.items():
            plot_roc_curve(y_test, y_proba, model_name=model_name, ax=ax)
        ax.set_title("ROC Curves - All Models", fontsize=16, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
        plt.tight_layout()
        roc_path = os.path.join(args.output_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"ROC curve saved to: {roc_path}")
        plt.close()

        # Score distributions - use histogram with log scale for better visibility
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # define colors for different models
        model_colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        
        # use more bins for better granularity
        bin_edges = np.linspace(0, 1, 101)  # 100 bins
        
        for idx, (model_name, (y_test, y_pred, y_proba)) in enumerate(all_results.items()):
            if y_proba.ndim > 1:
                y_proba_positive = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten()
            else:
                y_proba_positive = y_proba

            normal_scores = y_proba_positive[y_test == 0]
            fake_scores = y_proba_positive[y_test == 1]

            # plot histograms with different styles
            ax.hist(normal_scores, bins=bin_edges, alpha=0.6, 
                   label=f"{model_name} - Normal", 
                   color=model_colors[idx], edgecolor="black",
                   histtype='step', linewidth=2.5, linestyle='-')
            ax.hist(fake_scores, bins=bin_edges, alpha=0.6, 
                   label=f"{model_name} - Fake", 
                   color=model_colors[idx], edgecolor="black",
                   histtype='step', linewidth=2.5, linestyle='--')

        # add threshold line
        ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=2, alpha=0.7, zorder=0, label="Threshold (0.5)")
        
        ax.set_xlabel("Prediction Score (Fake Probability)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Count", fontsize=14, fontweight="bold")
        ax.set_title("Score Distributions - All Models", fontsize=16, fontweight="bold")
        ax.legend(fontsize=10, loc="upper left", framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3, linestyle="--", axis="both")
        ax.set_xlim([0, 1])
        ax.set_yscale('log')  # use log scale for y-axis to better see distributions
        
        plt.tight_layout()
        dist_path = os.path.join(args.output_dir, "score_distribution.png")
        plt.savefig(dist_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Score distribution saved to: {dist_path}")
        plt.close()

        # save metrics
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")

        # print summary
        print("\n" + "=" * 60)
        print("Evaluation Summary")
        print("=" * 60)
        print(f"{'Model':<25} {'AUC':<8} {'Precision':<10} {'Recall':<10} {'F1':<8}")
        print("-" * 60)
        for model_name, metrics in all_metrics.items():
            print(
                f"{model_name:<25} {metrics['auc']:<8.4f} {metrics['precision']:<10.4f} "
                f"{metrics['recall']:<10.4f} {metrics['f1']:<8.4f}"
            )

        print("\n" + "=" * 60)
        print("Evaluation completed successfully!")
        print("=" * 60)
        print(f"Outputs saved to: {args.output_dir}/")
    else:
        print("\nNo models were evaluated. Check that model files exist.")


if __name__ == "__main__":
    main()

