"""
Simple test script for FastAPI endpoints.

This script tests the basic functionality of the API:
- Health check
- Model loading
- Prediction endpoint
"""

import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://127.0.0.1:8000"


def test_health():
    """Test health check endpoint."""
    print("=" * 60)
    print("Testing /health endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"Status: {data.get('status')}")
        print(f"Model loaded: {data.get('model_loaded')}")
        print(f"Model type: {data.get('model_type', 'N/A')}")
        return data
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API. Is the server running?")
        print("Start the server with: python deployment/fastapi_server.py")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def load_model(model_path="models/baselines/logistic_regression.pkl", model_type="logistic_regression"):
    """Load a model via API."""
    print("\n" + "=" * 60)
    print("Loading model via /load_model endpoint")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{BASE_URL}/load_model",
            params={
                "model_path": model_path,
                "model_type": model_type,
                "threshold": 0.5
            }
        )
        response.raise_for_status()
        data = response.json()
        print(f"Status: {data.get('status')}")
        print(f"Message: {data.get('message')}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        return False


def test_predict():
    """Test prediction endpoint."""
    print("\n" + "=" * 60)
    print("Testing /predict endpoint")
    print("=" * 60)
    
    # create sample time series data
    start_date = datetime.now() - timedelta(days=2)
    time_series = []
    for i in range(48):  # 48 hours
        time_series.append({
            "timestamp": (start_date + timedelta(hours=i)).isoformat(),
            "views": 1000 + i * 10,
            "likes": 50 + i * 2,
            "comments": 10 + i,
            "shares": 5 + i,
        })
    
    request_data = {
        "id": "test_video_001",
        "time_series": time_series
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=request_data
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"ID: {data.get('id')}")
        print(f"Score: {data.get('score'):.4f}")
        print(f"Label: {data.get('label')}")
        print(f"Is Fake: {data.get('is_fake')}")
        print(f"Model Type: {data.get('model_type')}")
        print(f"Threshold: {data.get('threshold')}")
        
        return data
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP {e.response.status_code}")
        print(f"Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    """Run all tests."""
    print("FastAPI Server Test")
    print("=" * 60)
    print(f"Testing API at: {BASE_URL}")
    print("Make sure the server is running!")
    print("=" * 60)
    
    # test health
    health_data = test_health()
    if health_data is None:
        return
    
    # check if model is loaded
    if not health_data.get('model_loaded', False):
        print("\nModel not loaded. Attempting to load model...")
        if not load_model():
            print("Failed to load model. Cannot test predictions.")
            return
    
    # test prediction
    result = test_predict()
    if result:
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        print(f"\nExample response:")
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 60)
        print("Prediction test failed!")
        print("=" * 60)


if __name__ == "__main__":
    main()


