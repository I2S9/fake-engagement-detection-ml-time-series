"""
Main training script for fake engagement detection models.

This script orchestrates the complete training pipeline:
1. Load configuration
2. Generate/load data
3. Preprocess data
4. Extract features (for baselines)
5. Create sequences (for sequential models)
6. Train models (baselines and sequential)
7. Evaluate models
8. Log with MLflow
9. Save models

Usage:
    python src/training/train_main.py --config config/config.yaml
    python -m src.training.train_main --config config/config.yaml
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import os
import yaml

# add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.simulate_timeseries import generate_dataset
from src.data.preprocess import load_and_preprocess
from src.data.sequence_preparation import prepare_sequences_for_training
from src.data.dataset import create_dataloaders_from_dict
from src.features.temporal_features import extract_temporal_features
from src.training.train import (
    train_multiple_baselines,
    train_model_from_config,
)
from src.training.evaluate import (
    evaluate_sequential_model,
    compute_metrics,
)
from src.utils.config import load_config, update_config_with_data

# try to import MLflow
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available. Install with: pip install mlflow")


def setup_mlflow(config: Dict[str, Any], experiment_name: str = "fake_engagement_detection"):
    """Setup MLflow experiment."""
    if not MLFLOW_AVAILABLE:
        return None

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

    return mlflow.start_run()


def log_config_to_mlflow(config: Dict[str, Any]):
    """Log configuration to MLflow."""
    if not MLFLOW_AVAILABLE:
        return

    # flatten config for logging
    def flatten_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_config = flatten_dict(config)
    mlflow.log_params(flat_config)


def log_metrics_to_mlflow(metrics: Dict[str, float], step: Optional[int] = None):
    """Log metrics to MLflow."""
    if not MLFLOW_AVAILABLE:
        return

    mlflow.log_metrics(metrics, step=step)


def log_model_to_mlflow(model, model_name: str, artifact_path: Optional[str] = None):
    """Log model to MLflow."""
    if not MLFLOW_AVAILABLE:
        return

    if artifact_path is None:
        artifact_path = model_name

    if isinstance(model, nn.Module):
        mlflow.pytorch.log_model(model, artifact_path)
    else:
        # for sklearn models, use pickle
        import pickle
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(model, f)
            mlflow.log_artifact(f.name, artifact_path)
            os.unlink(f.name)


def ensure_data_exists(data_path: str, config: Dict[str, Any]):
    """Ensure data file exists, generate if not."""
    if os.path.exists(data_path):
        print(f"Data file found: {data_path}")
        return

    print(f"Data file not found: {data_path}")
    print("Generating synthetic data...")

    # get generation parameters from config if available
    data_config = config.get("data_generation", {})
    n_normal = data_config.get("n_normal", 100)
    n_fake = data_config.get("n_fake", 30)
    length_days = data_config.get("length_days", 30)
    frequency = data_config.get("frequency", "H")

    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    generate_dataset(
        n_normal=n_normal,
        n_fake=n_fake,
        length_days=length_days,
        frequency=frequency,
        output_path=data_path,
        output_format="parquet",
        random_seed=config.get("training", {}).get("random_seed", 42),
    )

    print(f"Data generated and saved to: {data_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train fake engagement detection models")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data file (overrides config)",
    )
    parser.add_argument(
        "--model-types",
        type=str,
        nargs="+",
        default=None,
        help="Model types to train (e.g., lstm tcn autoencoder)",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip baseline model training",
    )
    parser.add_argument(
        "--skip-sequential",
        action="store_true",
        help="Skip sequential model training",
    )

    args = parser.parse_args()

    # load configuration
    print("=" * 60)
    print("Loading configuration...")
    print("=" * 60)
    config = load_config(args.config)
    print(f"Configuration loaded from: {args.config}")

    # setup random seeds
    random_seed = config.get("training", {}).get("random_seed", 42)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # setup device
    device_config = config.get("training", {}).get("device", "cpu")
    if device_config == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(device_config if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # determine data path
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = config.get("data", {}).get("data_path", "data/raw/engagement.parquet")
        if not os.path.isabs(data_path):
            data_path = project_root / data_path

    # ensure data exists
    ensure_data_exists(str(data_path), config)

    # setup MLflow
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow_run = setup_mlflow(config)
        log_config_to_mlflow(config)
        print("MLflow experiment started")

    try:
        # load and preprocess data
        print("\n" + "=" * 60)
        print("Loading and preprocessing data...")
        print("=" * 60)
        df = load_and_preprocess(
            file_path=str(data_path),
            target_timezone="UTC",
            resample_frequency="h",  # use lowercase 'h' instead of 'H'
            handle_missing=True,
            missing_method="forward",
            normalize=False,
        )
        print(f"Data shape: {df.shape}")
        print(f"Number of videos: {df['id'].nunique()}")
        print(f"Label distribution:")
        print(df['label'].value_counts())

        # train baseline models
        if not args.skip_baselines:
            print("\n" + "=" * 60)
            print("Training baseline models...")
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
                # get labels per id (assuming one label per id)
                labels_per_id = df.groupby("id")["label"].first().reset_index()
                # ensure features_df has 'id' column
                if "id" not in features_df.columns:
                    # if features_df has index as id, reset it
                    if features_df.index.name == "id" or isinstance(features_df.index, pd.Index):
                        features_df = features_df.reset_index()
                if "id" in features_df.columns:
                    features_df = features_df.merge(labels_per_id, on="id", how="left")
                else:
                    print("Warning: Could not merge labels - 'id' column missing from features_df")
            
            # verify label is present
            if "label" not in features_df.columns:
                raise ValueError("Label column not found in features_df. Cannot train baseline models.")
            
            print(f"Features shape: {features_df.shape}")
            print(f"Feature columns: {len([c for c in features_df.columns if c not in ['id', 'label']])}")
            print(f"Label distribution in features:")
            print(features_df['label'].value_counts())

            # determine which baseline models to train
            baseline_models = config.get("training", {}).get("baseline_models", [
                "logistic_regression",
                "random_forest",
                "isolation_forest",
            ])

            # train baselines
            baseline_save_dir = config.get("training", {}).get("baseline_model_save_dir", "models/baselines")
            os.makedirs(baseline_save_dir, exist_ok=True)

            baseline_results = train_multiple_baselines(
                features_df=features_df,
                model_types=baseline_models,
                test_size=config.get("data", {}).get("test_size", 0.2),
                random_state=random_seed,
                save_dir=baseline_save_dir,
            )

            # evaluate and log baseline results
            print("\n" + "=" * 60)
            print("Evaluating baseline models...")
            print("=" * 60)
            for model_type, (model, X_test, y_test, y_pred, y_proba) in baseline_results.items():
                metrics = compute_metrics(y_test, y_pred, y_proba)
                print(f"\n{model_type} metrics:")
                for metric_name, metric_value in metrics.items():
                    print(f"  {metric_name}: {metric_value:.4f}")

                if MLFLOW_AVAILABLE:
                    log_metrics_to_mlflow({f"{model_type}_{k}": v for k, v in metrics.items()})
                    log_model_to_mlflow(model, f"{model_type}_baseline")

        # train sequential models
        if not args.skip_sequential:
            print("\n" + "=" * 60)
            print("Training sequential models...")
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
                random_state=random_seed,
            )

            # get input size from sequences
            X_train = sequences_dict["X_train"]
            input_size = X_train.shape[2]  # [n_sequences, seq_len, n_features]
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

            print(f"Train batches: {len(dataloaders['train'])}")
            print(f"Val batches: {len(dataloaders['val'])}")
            print(f"Test batches: {len(dataloaders['test'])}")
            print(f"Batch size: {batch_size}")

            # determine which sequential models to train
            if args.model_types:
                sequential_models = args.model_types
            else:
                sequential_models = config.get("training", {}).get("sequential_models", ["lstm", "tcn", "autoencoder"])

            # train sequential models
            sequential_save_dir = config.get("training", {}).get("model_save_dir", "models/sequential")
            os.makedirs(sequential_save_dir, exist_ok=True)

            for model_type in sequential_models:
                if model_type not in ["lstm", "tcn", "autoencoder"]:
                    print(f"Warning: Unknown model type {model_type}, skipping")
                    continue

                print(f"\n{'='*60}")
                print(f"Training {model_type.upper()}")
                print(f"{'='*60}")

                try:
                    # train model
                    model, history = train_model_from_config(
                        model_type=model_type,
                        dataloaders=dataloaders,
                        config=config,
                        device=device,
                        save_dir=sequential_save_dir,
                    )

                    # evaluate model
                    print(f"\nEvaluating {model_type}...")
                    test_metrics = evaluate_sequential_model(
                        model=model,
                        test_loader=dataloaders["test"],
                        device=device,
                        model_type=model_type,
                    )

                    print(f"\n{model_type} test metrics:")
                    for metric_name, metric_value in test_metrics.items():
                        print(f"  {metric_name}: {metric_value:.4f}")

                    # log to MLflow
                    if MLFLOW_AVAILABLE:
                        # log training history
                        for epoch, (train_loss, val_loss) in enumerate(
                            zip(history["train_loss"], history["val_loss"])
                        ):
                            log_metrics_to_mlflow(
                                {
                                    f"{model_type}_train_loss": train_loss,
                                    f"{model_type}_val_loss": val_loss,
                                },
                                step=epoch,
                            )

                        # log test metrics
                        log_metrics_to_mlflow(
                            {f"{model_type}_test_{k}": v for k, v in test_metrics.items()}
                        )

                        # log model
                        log_model_to_mlflow(model, f"{model_type}_sequential")

                except Exception as e:
                    print(f"Error training {model_type}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)

    finally:
        if mlflow_run is not None:
            mlflow.end_run()
            print("MLflow run ended")


if __name__ == "__main__":
    main()

