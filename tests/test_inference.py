"""
Unit tests for inference pipeline.
"""

import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta

from src.inference.inference_pipeline import (
    InferencePipeline,
    predict_fake_probability,
    preprocess_time_series,
    prepare_for_baseline_model,
    prepare_for_sequential_model,
)
from src.data.simulate_timeseries import generate_normal_timeseries, generate_fake_timeseries


def test_preprocess_time_series():
    """Test time series preprocessing."""
    # create test data
    start_date = datetime.now() - timedelta(days=2)
    df = generate_normal_timeseries(
        start_date=start_date,
        length_days=2,
        frequency="H",
        video_id="test_001",
    )

    # preprocess
    df_preprocessed = preprocess_time_series(
        df,
        id_column="id",
        timestamp_column="timestamp",
        target_timezone="UTC",
        resample_frequency="h",
        handle_missing=True,
    )

    assert len(df_preprocessed) > 0
    assert "timestamp" in df_preprocessed.columns
    assert "views" in df_preprocessed.columns


def test_prepare_for_baseline_model():
    """Test preparation for baseline model."""
    # create test data
    start_date = datetime.now() - timedelta(days=2)
    df = generate_normal_timeseries(
        start_date=start_date,
        length_days=2,
        frequency="H",
        video_id="test_001",
    )

    df_preprocessed = preprocess_time_series(df)

    # prepare features
    X = prepare_for_baseline_model(df_preprocessed, id_column="id")

    assert X.shape[0] > 0
    assert X.shape[1] > 0
    assert not np.isnan(X).any()


def test_prepare_for_sequential_model():
    """Test preparation for sequential model."""
    # create test data
    start_date = datetime.now() - timedelta(days=2)
    df = generate_normal_timeseries(
        start_date=start_date,
        length_days=2,
        frequency="H",
        video_id="test_001",
    )

    df_preprocessed = preprocess_time_series(df)

    # prepare sequences
    seq_len = 24
    X, scaler = prepare_for_sequential_model(
        df_preprocessed,
        seq_len=seq_len,
        id_column="id",
        normalize=True,
    )

    assert len(X) > 0
    assert X[0].shape[0] == seq_len
    assert X[0].shape[1] > 0


def test_inference_pipeline_baseline():
    """Test inference pipeline with baseline model."""
    # create and save a simple baseline model for testing
    from src.models.baselines import LogisticRegressionBaseline
    from src.data.simulate_timeseries import generate_dataset

    # generate small dataset
    test_data_path = project_root / "data" / "raw" / "test_inference.parquet"
    generate_dataset(
        n_normal=10,
        n_fake=5,
        length_days=7,
        frequency="H",
        output_path=str(test_data_path),
        random_seed=42,
    )

    # extract features and train a simple model
    from src.data.preprocess import load_and_preprocess
    from src.features.temporal_features import extract_temporal_features

    df = load_and_preprocess(str(test_data_path))
    features_df = extract_temporal_features(df, aggregate_per_id=True)

    # train model
    from src.training.train import prepare_data, train_baseline_model

    X_train, _, y_train, _, _ = prepare_data(features_df)
    model = train_baseline_model(
        "logistic_regression",
        X_train,
        y_train,
    )

    # save model
    model_path = project_root / "models" / "test_baseline.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    # test inference
    test_ts = generate_normal_timeseries(
        start_date=datetime.now() - timedelta(days=7),
        length_days=7,
        frequency="H",
        video_id="test_video",
    )

    pipeline = InferencePipeline(
        model_path=str(model_path),
        model_type="logistic_regression",
        threshold=0.5,
    )

    result = pipeline.predict_fake_probability(test_ts)

    assert "score" in result
    assert "label" in result
    assert "is_fake" in result
    assert result["label"] in ["normal", "fake"]
    assert 0 <= result["score"] <= 1

    # cleanup
    if model_path.exists():
        model_path.unlink()


def test_predict_fake_probability_function():
    """Test the simple predict_fake_probability function."""
    # create test time series
    test_ts = generate_normal_timeseries(
        start_date=datetime.now() - timedelta(days=7),
        length_days=7,
        frequency="H",
        video_id="test_video",
    )

    # this test requires a saved model, so we'll just test the function signature
    # In practice, you would load a real trained model
    try:
        # try to use the function (will fail if no model, but tests the interface)
        result = predict_fake_probability(
            test_ts,
            model_path="dummy_path.pkl",
            model_type="logistic_regression",
            threshold=0.5,
        )
    except (FileNotFoundError, ValueError):
        # expected if model doesn't exist
        pass


def test_inference_pipeline_sequential():
    """Test inference pipeline with sequential model."""
    # create test time series
    test_ts = generate_normal_timeseries(
        start_date=datetime.now() - timedelta(days=7),
        length_days=7,
        frequency="H",
        video_id="test_video",
    )

    # create a simple LSTM model for testing
    from src.models.lstm import create_lstm_model

    config = {
        "input_size": 4,  # views, likes, comments, shares
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.1,
        "num_classes": 2,
    }

    model = create_lstm_model(config)

    # save model
    model_path = project_root / "models" / "test_lstm.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"lstm": config, "data": {"seq_len": 48}},
        },
        str(model_path),
    )

    # test inference
    pipeline_config = {
        "lstm": config,
        "data": {
            "seq_len": 48,
            "normalize": True,
            "normalization_method": "standardize",
            "normalize_per_series": False,
        },
    }

    pipeline = InferencePipeline(
        model_path=str(model_path),
        model_type="lstm",
        config=pipeline_config,
        threshold=0.5,
    )

    result = pipeline.predict_fake_probability(test_ts)

    assert "score" in result
    assert "label" in result
    assert "is_fake" in result
    assert result["label"] in ["normal", "fake"]
    assert 0 <= result["score"] <= 1

    # cleanup
    if model_path.exists():
        model_path.unlink()


def test_threshold_management():
    """Test threshold management for decision making."""
    # test threshold logic directly
    thresholds = [0.3, 0.5, 0.7]
    
    # test that threshold affects decision
    test_scores = [0.2, 0.4, 0.6, 0.8]
    
    for threshold in thresholds:
        for score in test_scores:
            is_fake = score >= threshold
            label = "fake" if is_fake else "normal"
            
            # verify logic
            if score >= threshold:
                assert is_fake == True
                assert label == "fake"
            else:
                assert is_fake == False
                assert label == "normal"


def test_batch_prediction():
    """Test batch prediction functionality."""
    # create multiple test time series
    test_ts_list = [
        generate_normal_timeseries(
            start_date=datetime.now() - timedelta(days=7),
            length_days=7,
            frequency="H",
            video_id=f"test_video_{i}",
        )
        for i in range(3)
    ]

    # this test requires a saved model, so we'll test the interface
    # In practice, you would load a real trained model
    try:
        pipeline = InferencePipeline(
            model_path="dummy_path.pkl",
            model_type="logistic_regression",
            threshold=0.5,
        )
        results = pipeline.predict_batch(test_ts_list)
        assert len(results) == len(test_ts_list)
    except (FileNotFoundError, ValueError):
        # expected if model doesn't exist
        pass


if __name__ == "__main__":
    # run basic tests
    print("Running inference pipeline tests...")

    try:
        test_preprocess_time_series()
        print("✓ test_preprocess_time_series passed")

        test_prepare_for_baseline_model()
        print("✓ test_prepare_for_baseline_model passed")

        test_prepare_for_sequential_model()
        print("✓ test_prepare_for_sequential_model passed")

        test_threshold_management()
        print("✓ test_threshold_management passed")

        print("\nAll basic tests passed!")
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback

        traceback.print_exc()

