"""
Example usage of the inference pipeline.

This script demonstrates how to use the inference pipeline to predict
fake engagement probability for new time series data.
"""

import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
from src.inference.inference_pipeline import InferencePipeline, predict_fake_probability
from src.data.simulate_timeseries import generate_normal_timeseries, generate_fake_timeseries
from src.utils.config import load_config


def example_baseline_model():
    """Example using a baseline model."""
    print("=" * 60)
    print("Example: Baseline Model Inference")
    print("=" * 60)

    # create test time series
    test_ts = generate_normal_timeseries(
        start_date=datetime.now() - timedelta(days=7),
        length_days=7,
        frequency="H",
        video_id="example_video_001",
    )

    print(f"\nTime series created: {len(test_ts)} time steps")
    print(f"Columns: {test_ts.columns.tolist()}")

    # example usage (requires a trained model)
    # model_path = "models/baselines/logistic_regression.pkl"
    # result = predict_fake_probability(
    #     test_ts,
    #     model_path=model_path,
    #     model_type="logistic_regression",
    #     threshold=0.5,
    # )
    # print(f"\nPrediction result:")
    # print(f"  Score: {result['score']:.4f}")
    # print(f"  Label: {result['label']}")
    # print(f"  Is Fake: {result['is_fake']}")

    print("\nNote: Uncomment the code above and provide a trained model path to run.")


def example_sequential_model():
    """Example using a sequential model (LSTM/TCN/Autoencoder)."""
    print("\n" + "=" * 60)
    print("Example: Sequential Model Inference")
    print("=" * 60)

    # create test time series
    test_ts = generate_normal_timeseries(
        start_date=datetime.now() - timedelta(days=7),
        length_days=7,
        frequency="H",
        video_id="example_video_002",
    )

    print(f"\nTime series created: {len(test_ts)} time steps")

    # load config
    try:
        config = load_config()
        print("Configuration loaded successfully")
    except FileNotFoundError:
        print("Config file not found. Using default values.")
        config = None

    # example usage (requires a trained model)
    # model_path = "models/sequential/lstm_best.pth"
    # pipeline = InferencePipeline(
    #     model_path=model_path,
    #     model_type="lstm",
    #     config=config,
    #     threshold=0.5,
    # )
    #
    # result = pipeline.predict_fake_probability(test_ts)
    # print(f"\nPrediction result:")
    # print(f"  Score: {result['score']:.4f}")
    # print(f"  Label: {result['label']}")
    # print(f"  Is Fake: {result['is_fake']}")

    print("\nNote: Uncomment the code above and provide a trained model path to run.")


def example_batch_prediction():
    """Example of batch prediction."""
    print("\n" + "=" * 60)
    print("Example: Batch Prediction")
    print("=" * 60)

    # create multiple test time series
    test_ts_list = [
        generate_normal_timeseries(
            start_date=datetime.now() - timedelta(days=7),
            length_days=7,
            frequency="H",
            video_id=f"example_video_{i:03d}",
        )
        for i in range(3)
    ]

    print(f"\nCreated {len(test_ts_list)} time series for batch prediction")

    # example usage (requires a trained model)
    # model_path = "models/baselines/random_forest.pkl"
    # pipeline = InferencePipeline(
    #     model_path=model_path,
    #     model_type="random_forest",
    #     threshold=0.5,
    # )
    #
    # results = pipeline.predict_batch(test_ts_list)
    # print(f"\nBatch prediction results:")
    # for i, result in enumerate(results):
    #     print(f"  Video {i+1}: Score={result['score']:.4f}, Label={result['label']}")

    print("\nNote: Uncomment the code above and provide a trained model path to run.")


def example_threshold_adjustment():
    """Example showing how threshold affects predictions."""
    print("\n" + "=" * 60)
    print("Example: Threshold Adjustment")
    print("=" * 60)

    # create test time series
    test_ts = generate_normal_timeseries(
        start_date=datetime.now() - timedelta(days=7),
        length_days=7,
        frequency="H",
        video_id="example_video_003",
    )

    print("\nDifferent thresholds produce different decisions:")
    print("  Lower threshold (0.3): More sensitive, flags more as fake")
    print("  Default threshold (0.5): Balanced")
    print("  Higher threshold (0.7): Less sensitive, flags fewer as fake")

    # example usage
    # model_path = "models/baselines/logistic_regression.pkl"
    # for threshold in [0.3, 0.5, 0.7]:
    #     result = predict_fake_probability(
    #         test_ts,
    #         model_path=model_path,
    #         model_type="logistic_regression",
    #         threshold=threshold,
    #     )
    #     print(f"\n  Threshold {threshold}: {result['label']} (score: {result['score']:.4f})")

    print("\nNote: Uncomment the code above and provide a trained model path to run.")


if __name__ == "__main__":
    example_baseline_model()
    example_sequential_model()
    example_batch_prediction()
    example_threshold_adjustment()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nThe inference pipeline provides:")
    print("  1. Simple function: predict_fake_probability()")
    print("  2. Pipeline class: InferencePipeline for advanced usage")
    print("  3. Batch prediction support")
    print("  4. Configurable thresholds")
    print("  5. Support for both baseline and sequential models")
    print("\nTo use:")
    print("  1. Train a model using notebooks/03_modeling.ipynb")
    print("  2. Load the model using InferencePipeline")
    print("  3. Call predict_fake_probability() with new time series data")

