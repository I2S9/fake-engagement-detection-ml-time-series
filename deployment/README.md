# FastAPI Server for Fake Engagement Detection

This directory contains the FastAPI server for serving fake engagement detection predictions.

## Setup

1. Install dependencies:
```bash
pip install fastapi uvicorn
```

2. Set environment variables (optional):
```bash
export MODEL_PATH="models/baselines/random_forest.pkl"
export MODEL_TYPE="random_forest"
export CONFIG_PATH="config/config.yaml"
export PORT=8000
export HOST="0.0.0.0"
```

## Running the Server

### Option 1: Direct execution
```bash
python deployment/fastapi_server.py
```

### Option 2: Using uvicorn
```bash
uvicorn deployment.fastapi_server:app --host 0.0.0.0 --port 8000
```

### Option 3: With reload (development)
```bash
uvicorn deployment.fastapi_server:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### GET /
Root endpoint with API information.

### GET /health
Health check endpoint. Returns API status and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "random_forest",
  "model_path": "models/baselines/random_forest.pkl"
}
```

### POST /predict
Predict fake engagement probability for a single time series.

**Request:**
```json
{
  "id": "video_001",
  "time_series": [
    {
      "timestamp": "2024-01-01T00:00:00",
      "views": 1000,
      "likes": 50,
      "comments": 10,
      "shares": 5
    },
    ...
  ]
}
```

**Response:**
```json
{
  "id": "video_001",
  "score": 0.2345,
  "label": "normal",
  "is_fake": false,
  "model_type": "random_forest",
  "threshold": 0.5
}
```

### POST /predict_batch
Predict fake engagement probability for multiple time series.

**Request:**
```json
{
  "time_series_list": [
    {
      "id": "video_001",
      "time_series": [...]
    },
    {
      "id": "video_002",
      "time_series": [...]
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "id": "video_001",
      "score": 0.2345,
      "label": "normal",
      "is_fake": false,
      "model_type": "random_forest",
      "threshold": 0.5
    },
    ...
  ],
  "total": 2,
  "successful": 2,
  "failed": 0
}
```

### POST /load_model
Load a model for inference.

**Query Parameters:**
- `model_path` (required): Path to model file
- `model_type` (optional): Type of model (auto-detected if not provided)
- `config_path` (optional): Path to config file
- `threshold` (optional): Decision threshold (default: 0.5)

**Example:**
```bash
curl -X POST "http://localhost:8000/load_model?model_path=models/baselines/random_forest.pkl&model_type=random_forest&threshold=0.5"
```

## Testing

Run the test script:
```bash
python deployment/test_api.py
```

Or test manually with curl:

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test_video",
    "time_series": [
      {
        "timestamp": "2024-01-01T00:00:00",
        "views": 1000,
        "likes": 50,
        "comments": 10,
        "shares": 5
      }
    ]
  }'
```

## Error Handling

The API handles various error cases:

- **503 Service Unavailable**: Model not loaded
- **400 Bad Request**: Invalid data format, missing columns, negative values
- **422 Unprocessable Entity**: Validation errors (invalid timestamp format, etc.)
- **500 Internal Server Error**: Unexpected errors during prediction

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Production Deployment

For production, consider:

1. Using a production ASGI server like Gunicorn with Uvicorn workers:
```bash
gunicorn deployment.fastapi_server:app -w 4 -k uvicorn.workers.UvicornWorker
```

2. Adding authentication/authorization
3. Setting up proper logging
4. Using environment variables for configuration
5. Adding rate limiting
6. Setting up monitoring and health checks
