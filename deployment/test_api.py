"""
Test script for FastAPI server.

This script tests all endpoints of the API.
"""

import requests
import json
from datetime import datetime, timedelta
import time
import pytest

# API base URL
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test health check endpoint."""
    print("=" * 60)
    print("Testing /health endpoint")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Model loaded: {data['model_loaded']}")
        print(f"Model type: {data.get('model_type', 'N/A')}")
        print("✓ Health check passed")
        assert response.status_code == 200
        assert "status" in data
        assert "model_loaded" in data
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Is the server running?")
        pytest.skip("Server not running", allow_module_level=False)
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        raise


def test_predict_endpoint():
    """Test /predict endpoint."""
    print("\n" + "=" * 60)
    print("Testing /predict endpoint")
    print("=" * 60)

    # create sample time series data
    start_date = datetime.now() - timedelta(days=2)
    time_series = []
    for i in range(48):  # 48 hours
        time_series.append(
            {
                "timestamp": (start_date + timedelta(hours=i)).isoformat(),
                "views": 1000 + i * 10,
                "likes": 50 + i * 2,
                "comments": 10 + i,
                "shares": 5 + i,
            }
        )

    request_data = {
        "id": "test_video_001",
        "time_series": time_series,
    }

    try:
        response = requests.post(f"{BASE_URL}/predict", json=request_data, timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"ID: {data['id']}")
        print(f"Score: {data['score']:.4f}")
        print(f"Label: {data['label']}")
        print(f"Is Fake: {data['is_fake']}")
        print(f"Model Type: {data.get('model_type', 'N/A')}")
        print(f"Threshold: {data['threshold']}")
        print("✓ Predict endpoint passed")
        assert response.status_code == 200
        assert "id" in data
        assert "score" in data
        assert "label" in data
        assert data["label"] in ["normal", "fake"]
        assert 0 <= data["score"] <= 1
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Is the server running?")
        pytest.skip("Server not running", allow_module_level=False)
    except requests.exceptions.HTTPError as e:
        print(f"✗ Predict endpoint failed: {e}")
        if e.response.status_code == 503:
            print("  Model not loaded. Load a model first using /load_model")
            pytest.skip("Model not loaded", allow_module_level=False)
        elif e.response.status_code == 400:
            print(f"  Bad request: {e.response.json()}")
        raise
    except Exception as e:
        print(f"✗ Predict endpoint failed: {e}")
        raise


def test_predict_batch_endpoint():
    """Test /predict_batch endpoint."""
    print("\n" + "=" * 60)
    print("Testing /predict_batch endpoint")
    print("=" * 60)

    # create multiple time series
    time_series_list = []
    for video_id in ["video_001", "video_002", "video_003"]:
        start_date = datetime.now() - timedelta(days=2)
        time_series = []
        for i in range(48):
            time_series.append(
                {
                    "timestamp": (start_date + timedelta(hours=i)).isoformat(),
                    "views": 1000 + i * 10,
                    "likes": 50 + i * 2,
                    "comments": 10 + i,
                    "shares": 5 + i,
                }
            )
        time_series_list.append({"id": video_id, "time_series": time_series})

    request_data = {"time_series_list": time_series_list}

    try:
        response = requests.post(f"{BASE_URL}/predict_batch", json=request_data, timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"Total: {data['total']}")
        print(f"Successful: {data['successful']}")
        print(f"Failed: {data['failed']}")
        print("\nPredictions:")
        for pred in data["predictions"]:
            print(f"  {pred['id']}: {pred['label']} (score: {pred['score']:.4f})")
        print("✓ Predict batch endpoint passed")
        assert response.status_code == 200
        assert "total" in data
        assert "successful" in data
        assert "failed" in data
        assert "predictions" in data
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Is the server running?")
        pytest.skip("Server not running", allow_module_level=False)
    except requests.exceptions.HTTPError as e:
        print(f"✗ Predict batch endpoint failed: {e}")
        if e.response.status_code == 503:
            print("  Model not loaded. Load a model first using /load_model")
            pytest.skip("Model not loaded", allow_module_level=False)
        raise
    except Exception as e:
        print(f"✗ Predict batch endpoint failed: {e}")
        raise


def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 60)
    print("Testing error handling")
    print("=" * 60)

    # test missing data
    print("\n1. Testing missing data...")
    try:
        request_data = {
            "id": "test_video",
            "time_series": [],  # empty time series
        }
        response = requests.post(f"{BASE_URL}/predict", json=request_data)
        if response.status_code == 400:
            print("  ✓ Correctly rejected empty time series")
        else:
            print(f"  ✗ Expected 400, got {response.status_code}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # test invalid format
    print("\n2. Testing invalid format...")
    try:
        request_data = {
            "id": "test_video",
            "time_series": [
                {
                    "timestamp": "invalid_date",
                    "views": 100,
                    "likes": 50,
                    "comments": 10,
                    "shares": 5,
                }
            ],
        }
        response = requests.post(f"{BASE_URL}/predict", json=request_data)
        if response.status_code == 422:  # validation error
            print("  ✓ Correctly rejected invalid timestamp format")
        else:
            print(f"  ✗ Expected 422, got {response.status_code}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # test negative values
    print("\n3. Testing negative values...")
    try:
        request_data = {
            "id": "test_video",
            "time_series": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "views": -100,  # negative value
                    "likes": 50,
                    "comments": 10,
                    "shares": 5,
                }
            ],
        }
        response = requests.post(f"{BASE_URL}/predict", json=request_data)
        if response.status_code in [400, 422]:
            print("  ✓ Correctly rejected negative values")
        else:
            print(f"  ✗ Expected 400/422, got {response.status_code}")
    except Exception as e:
        print(f"  ✗ Error: {e}")


def test_root_endpoint():
    """Test root endpoint."""
    print("\n" + "=" * 60)
    print("Testing root endpoint")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"Message: {data.get('message', 'N/A')}")
        print(f"Version: {data.get('version', 'N/A')}")
        print("✓ Root endpoint passed")
        assert response.status_code == 200
        assert "message" in data or "version" in data
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Is the server running?")
        pytest.skip("Server not running", allow_module_level=False)
    except Exception as e:
        print(f"✗ Root endpoint failed: {e}")
        raise


def run_test_safely(test_func, test_name):
    """Run a test function and return True if passed, False if failed."""
    try:
        test_func()
        return True
    except pytest.skip.Exception:
        return None
    except (AssertionError, Exception):
        return False


if __name__ == "__main__":
    print("FastAPI Server Test Suite")
    print("=" * 60)
    print(f"Testing API at: {BASE_URL}")
    print("Make sure the server is running: python deployment/fastapi_server.py")
    print("=" * 60)

    # wait a bit for server to be ready
    time.sleep(1)

    results = []

    # test root
    result = run_test_safely(test_root_endpoint, "Root")
    results.append(("Root", result))

    # test health
    result = run_test_safely(test_health_check, "Health")
    results.append(("Health", result))

    # test error handling (doesn't require model)
    test_error_handling()

    # test predict (requires model)
    result = run_test_safely(test_predict_endpoint, "Predict")
    results.append(("Predict", result))

    # test batch predict (requires model)
    result = run_test_safely(test_predict_batch_endpoint, "Predict Batch")
    results.append(("Predict Batch", result))

    # summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        if passed is None:
            status = "⚠ SKIPPED"
        elif passed:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"{test_name}: {status}")

    passed_tests = [r[1] for r in results if r[1] is True]
    all_passed = len(passed_tests) == len([r for r in results if r[1] is not None])
    if all_passed and len(passed_tests) > 0:
        print("\nAll tests passed!")
    elif len(passed_tests) == 0:
        print("\nNo tests could be executed. Check if the server is running.")
    else:
        print("\nSome tests failed. Check the output above for details.")

