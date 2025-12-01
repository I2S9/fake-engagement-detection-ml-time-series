"""
FastAPI server for fake engagement detection.

This module provides a REST API to serve predictions from trained models.
"""

import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

from src.inference.inference_pipeline import InferencePipeline
from src.utils.config import load_config

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create FastAPI app
app = FastAPI(
    title="Fake Engagement Detection API",
    description="API for detecting fake engagement in time series data",
    version="1.0.0",
)

# add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global variables for model
pipeline: Optional[InferencePipeline] = None
model_loaded = False


# Pydantic models for request/response
class TimeSeriesPoint(BaseModel):
    """Single time series data point."""

    timestamp: str
    views: float = Field(..., ge=0, description="Number of views")
    likes: float = Field(..., ge=0, description="Number of likes")
    comments: float = Field(..., ge=0, description="Number of comments")
    shares: float = Field(..., ge=0, description="Number of shares")

    @validator("timestamp")
    def validate_timestamp(cls, v):
        """Validate timestamp format."""
        try:
            pd.to_datetime(v)
            return v
        except Exception:
            raise ValueError("Invalid timestamp format. Use ISO format or standard datetime string.")


class TimeSeriesRequest(BaseModel):
    """Request model for single time series prediction."""

    id: str = Field(..., description="Video or user ID")
    time_series: List[TimeSeriesPoint] = Field(..., min_items=1, description="Time series data points")

    @validator("time_series")
    def validate_time_series_length(cls, v):
        """Validate time series has minimum length."""
        if len(v) < 1:
            raise ValueError("Time series must contain at least 1 data point")
        return v


class BatchTimeSeriesRequest(BaseModel):
    """Request model for batch prediction."""

    time_series_list: List[TimeSeriesRequest] = Field(..., min_items=1, description="List of time series")

    @validator("time_series_list")
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if len(v) == 0:
            raise ValueError("Batch must contain at least 1 time series")
        return v


class PredictionResponse(BaseModel):
    """Response model for prediction."""

    id: str
    score: float = Field(..., ge=0, le=1, description="Fake probability score")
    label: str = Field(..., description="Predicted label: 'normal' or 'fake'")
    is_fake: bool = Field(..., description="Boolean indicating if engagement is fake")
    model_type: Optional[str] = Field(None, description="Type of model used")
    threshold: float = Field(..., description="Threshold used for decision")


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""

    predictions: List[PredictionResponse]
    total: int
    successful: int
    failed: int


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    model_path: Optional[str] = None


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str
    detail: Optional[str] = None


def load_model(
    model_path: Optional[str] = None,
    model_type: Optional[str] = None,
    config_path: Optional[str] = None,
    threshold: float = 0.5,
) -> InferencePipeline:
    """
    Load the inference model.

    Parameters
    ----------
    model_path : str, optional
        Path to model file. If None, tries to load from environment or default
    model_type : str, optional
        Type of model. If None, tries to infer from path
    config_path : str, optional
        Path to config file
    threshold : float
        Decision threshold

    Returns
    -------
    InferencePipeline
        Loaded inference pipeline
    """
    global pipeline, model_loaded

    # try to get from environment variables
    if model_path is None:
        model_path = os.getenv("MODEL_PATH", None)

    if model_type is None:
        model_type = os.getenv("MODEL_TYPE", None)

    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", None)

    if model_path is None:
        raise ValueError(
            "Model path not provided. Set MODEL_PATH environment variable "
            "or provide model_path parameter."
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # infer model type from path if not provided
    if model_type is None:
        if model_path.endswith(".pkl"):
            # try to infer from filename
            if "lstm" in model_path.lower():
                model_type = "lstm"
            elif "tcn" in model_path.lower():
                model_type = "tcn"
            elif "autoencoder" in model_path.lower():
                model_type = "autoencoder"
            else:
                # default to baseline
                model_type = "logistic_regression"
        elif model_path.endswith(".pth"):
            # PyTorch model
            if "lstm" in model_path.lower():
                model_type = "lstm"
            elif "tcn" in model_path.lower():
                model_type = "tcn"
            elif "autoencoder" in model_path.lower():
                model_type = "autoencoder"
            else:
                model_type = "lstm"  # default
        else:
            raise ValueError("Cannot infer model type from path. Please specify model_type.")

    # load config if provided
    config = None
    if config_path and os.path.exists(config_path):
        try:
            config = load_config(config_path)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")

    # create pipeline
    try:
        pipeline = InferencePipeline(
            model_path=model_path,
            model_type=model_type,
            config=config,
            threshold=threshold,
        )
        model_loaded = True
        logger.info(f"Model loaded successfully: {model_path} (type: {model_type})")
        return pipeline
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def convert_request_to_dataframe(request: TimeSeriesRequest) -> pd.DataFrame:
    """
    Convert API request to DataFrame.

    Parameters
    ----------
    request : TimeSeriesRequest
        API request

    Returns
    -------
    pd.DataFrame
        DataFrame with time series data
    """
    data = []
    for point in request.time_series:
        data.append(
            {
                "id": request.id,
                "timestamp": point.timestamp,
                "views": point.views,
                "likes": point.likes,
                "comments": point.comments,
                "shares": point.shares,
            }
        )

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global pipeline, model_loaded

    try:
        model_path = os.getenv("MODEL_PATH", None)
        if model_path:
            load_model(model_path=model_path)
            logger.info("Model loaded on startup")
        else:
            logger.warning("MODEL_PATH not set. Model will need to be loaded manually.")
    except Exception as e:
        logger.error(f"Could not load model on startup: {e}")
        logger.info("API will start without model. Use /load_model endpoint to load.")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Fake Engagement Detection API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns
    -------
    HealthResponse
        Health status of the API
    """
    global pipeline, model_loaded

    status = "healthy" if model_loaded else "model_not_loaded"
    model_type = None
    model_path = None

    if pipeline is not None:
        model_type = pipeline.model_type
        model_path = pipeline.model_path

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        model_type=model_type,
        model_path=model_path,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TimeSeriesRequest):
    """
    Predict fake engagement probability for a single time series.

    Parameters
    ----------
    request : TimeSeriesRequest
        Time series data

    Returns
    -------
    PredictionResponse
        Prediction result with score and label
    """
    global pipeline, model_loaded

    if not model_loaded or pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please load model first.")

    try:
        # convert request to DataFrame
        df = convert_request_to_dataframe(request)

        # validate data
        if df.empty:
            raise HTTPException(status_code=400, detail="Empty time series data")

        required_columns = ["views", "likes", "comments", "shares"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, detail=f"Missing required columns: {missing_columns}"
            )

        # check for negative values
        numeric_cols = ["views", "likes", "comments", "shares"]
        if (df[numeric_cols] < 0).any().any():
            raise HTTPException(status_code=400, detail="Negative values not allowed in metrics")

        # make prediction
        result = pipeline.predict_fake_probability(df)

        return PredictionResponse(
            id=request.id,
            score=result["score"],
            label=result["label"],
            is_fake=result["is_fake"],
            model_type=pipeline.model_type,
            threshold=pipeline.threshold,
        )

    except ValueError as e:
        logger.error(f"ValueError in prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchTimeSeriesRequest):
    """
    Predict fake engagement probability for multiple time series.

    Parameters
    ----------
    request : BatchTimeSeriesRequest
        List of time series data

    Returns
    -------
    BatchPredictionResponse
        Batch prediction results
    """
    global pipeline, model_loaded

    if not model_loaded or pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please load model first.")

    predictions = []
    successful = 0
    failed = 0

    for ts_request in request.time_series_list:
        try:
            # convert request to DataFrame
            df = convert_request_to_dataframe(ts_request)

            # validate data
            if df.empty:
                predictions.append(
                    PredictionResponse(
                        id=ts_request.id,
                        score=0.0,
                        label="error",
                        is_fake=False,
                        model_type=pipeline.model_type,
                        threshold=pipeline.threshold,
                    )
                )
                failed += 1
                continue

            # make prediction
            result = pipeline.predict_fake_probability(df)

            predictions.append(
                PredictionResponse(
                    id=ts_request.id,
                    score=result["score"],
                    label=result["label"],
                    is_fake=result["is_fake"],
                    model_type=pipeline.model_type,
                    threshold=pipeline.threshold,
                )
            )
            successful += 1

        except Exception as e:
            logger.error(f"Error processing time series {ts_request.id}: {e}")
            predictions.append(
                PredictionResponse(
                    id=ts_request.id,
                    score=0.0,
                    label="error",
                    is_fake=False,
                    model_type=pipeline.model_type,
                    threshold=pipeline.threshold,
                )
            )
            failed += 1

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(request.time_series_list),
        successful=successful,
        failed=failed,
    )


@app.post("/load_model")
async def load_model_endpoint(
    model_path: str,
    model_type: Optional[str] = None,
    config_path: Optional[str] = None,
    threshold: float = 0.5,
):
    """
    Load a model for inference.

    Parameters
    ----------
    model_path : str
        Path to model file
    model_type : str, optional
        Type of model (auto-detected if not provided)
    config_path : str, optional
        Path to config file
    threshold : float
        Decision threshold

    Returns
    -------
    dict
        Status message
    """
    global pipeline, model_loaded

    try:
        load_model(
            model_path=model_path,
            model_type=model_type,
            config_path=config_path,
            threshold=threshold,
        )
        return {"status": "success", "message": f"Model loaded from {model_path}"}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(app, host=host, port=port)

