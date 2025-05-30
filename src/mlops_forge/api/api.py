"""API endpoints for model serving."""

import logging
import time
import uuid
from typing import Dict, List, Optional, Union

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from mlops_forge.api.schemas import (
    PredictionInput,
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
    HealthResponse,
    ModelMetadata,
)
from mlops_forge.config.settings import settings
from mlops_forge.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for serving machine learning models",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Prometheus metrics
PREDICTION_COUNT = Counter(
    "prediction_count_total", "Total number of predictions", ["model_version", "result"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency in seconds", ["model_version"]
)

# Global variables
MODEL = None
MODEL_VERSION = None
MODEL_FEATURES = None
MODEL_TYPE = None
MODEL_CREATED_AT = None
MODEL_METRICS = None


@app.on_event("startup")
async def startup_event():
    """Load model and other resources on startup."""
    global MODEL, MODEL_VERSION, MODEL_FEATURES, MODEL_TYPE, MODEL_CREATED_AT, MODEL_METRICS
    
    logger.info("Starting API server")
    logger.info(f"Environment: {settings.ENV}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    try:
        # In a real application, you would load the model from the model registry
        # For now, we'll use placeholders
        logger.info("Loading model...")
        # MODEL = get_model_trainer().load_model(settings.MODEL.MODEL_NAME)
        MODEL = "dummy_model"  # Placeholder for actual model
        
        # Set model metadata
        MODEL_VERSION = settings.MODEL.MODEL_VERSION
        MODEL_FEATURES = ["feature_1", "feature_2", "feature_3"]  # Example features
        MODEL_TYPE = "random_forest"  # Example model type
        MODEL_CREATED_AT = "2025-05-30T12:00:00Z"  # Example timestamp
        MODEL_METRICS = {  # Example metrics
            "accuracy": 0.95,
            "f1": 0.94,
            "precision": 0.93,
            "recall": 0.92
        }
        
        logger.info(f"Model loaded successfully: {MODEL_VERSION}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Continue without model for health check endpoints
        pass


@app.on_event("shutdown")
async def shutdown_event():
    """Release resources on shutdown."""
    logger.info("Shutting down API server")
    # Clean up resources if needed


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if MODEL is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "model_version": "unknown", "api_version": settings.VERSION},
        )
    
    return HealthResponse(
        status="ok",
        model_version=MODEL_VERSION,
        api_version=settings.VERSION,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metadata", response_model=ModelMetadata)
async def get_model_metadata():
    """Get model metadata."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelMetadata(
        model_name=settings.MODEL.MODEL_NAME,
        model_version=MODEL_VERSION,
        model_type=MODEL_TYPE,
        created_at=MODEL_CREATED_AT,
        features=MODEL_FEATURES,
        metrics=MODEL_METRICS,
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(
    data: PredictionInput, background_tasks: BackgroundTasks
) -> PredictionOutput:
    """Make a single prediction.
    
    Args:
        data: Prediction input data
        background_tasks: Background tasks for async operations
    
    Returns:
        Prediction output
    
    Raises:
        HTTPException: If prediction fails
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Generate a unique ID for this prediction
    prediction_id = str(uuid.uuid4())
    
    # Start timing
    start_time = time.time()
    
    try:
        # Validate input features
        for feature in MODEL_FEATURES:
            if feature not in data.features:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Create a DataFrame from the input features
        features_df = pd.DataFrame([data.features])
        
        # In a real application, you would use the model to make predictions
        # For now, we'll return a dummy prediction
        # prediction = MODEL.predict(features_df)[0]
        # probability = MODEL.predict_proba(features_df)[0][1] if hasattr(MODEL, "predict_proba") else None
        
        # Dummy prediction for illustration
        prediction = 1
        probability = 0.95
        
        # Record metrics
        prediction_result = "success"
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        prediction_result = "error"
        
        # Log error for monitoring
        background_tasks.add_task(
            log_error,
            error=str(e),
            input_data=data.features,
            prediction_id=prediction_id,
        )
        
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
    
    finally:
        # Record metrics
        latency = time.time() - start_time
        PREDICTION_LATENCY.labels(model_version=MODEL_VERSION).observe(latency)
        PREDICTION_COUNT.labels(model_version=MODEL_VERSION, result=prediction_result).inc()
    
    # Log prediction for monitoring (in background)
    background_tasks.add_task(
        log_prediction,
        input_data=data.features,
        prediction=prediction,
        probability=probability,
        prediction_id=prediction_id,
        latency=latency,
    )
    
    return PredictionOutput(
        prediction=prediction,
        probability=probability,
        prediction_id=prediction_id,
        model_version=MODEL_VERSION,
    )


@app.post("/batch-predict", response_model=BatchPredictionOutput)
async def batch_predict(
    data: BatchPredictionInput, background_tasks: BackgroundTasks
) -> BatchPredictionOutput:
    """Make batch predictions.
    
    Args:
        data: Batch prediction input data
        background_tasks: Background tasks for async operations
    
    Returns:
        Batch prediction output
    
    Raises:
        HTTPException: If prediction fails
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not data.instances:
        raise HTTPException(status_code=400, detail="No instances provided")
    
    # Generate unique IDs for each prediction
    prediction_ids = [str(uuid.uuid4()) for _ in range(len(data.instances))]
    
    # Start timing
    start_time = time.time()
    
    try:
        # Validate input features for each instance
        for i, instance in enumerate(data.instances):
            for feature in MODEL_FEATURES:
                if feature not in instance:
                    raise ValueError(f"Missing required feature '{feature}' in instance {i}")
        
        # Create a DataFrame from the input features
        features_df = pd.DataFrame(data.instances)
        
        # In a real application, you would use the model to make predictions
        # For now, we'll return dummy predictions
        # predictions = MODEL.predict(features_df).tolist()
        # probabilities = MODEL.predict_proba(features_df)[:, 1].tolist() if hasattr(MODEL, "predict_proba") else None
        
        # Dummy predictions for illustration
        predictions = [1, 0] * (len(data.instances) // 2) + ([1] if len(data.instances) % 2 else [])
        probabilities = [0.95, 0.25] * (len(data.instances) // 2) + ([0.8] if len(data.instances) % 2 else [])
        
        # Record metrics
        prediction_result = "success"
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        prediction_result = "error"
        
        # Log error for monitoring
        background_tasks.add_task(
            log_error,
            error=str(e),
            input_data=data.instances,
            prediction_id=prediction_ids[0],
        )
        
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")
    
    finally:
        # Record metrics
        latency = time.time() - start_time
        PREDICTION_LATENCY.labels(model_version=MODEL_VERSION).observe(latency)
        PREDICTION_COUNT.labels(model_version=MODEL_VERSION, result=prediction_result).inc(len(data.instances))
    
    # Log predictions for monitoring (in background)
    for i, (instance, prediction, probability, prediction_id) in enumerate(
        zip(data.instances, predictions, probabilities, prediction_ids)
    ):
        background_tasks.add_task(
            log_prediction,
            input_data=instance,
            prediction=prediction,
            probability=probability,
            prediction_id=prediction_id,
            latency=latency / len(data.instances),  # Average latency per instance
        )
    
    return BatchPredictionOutput(
        predictions=predictions,
        probabilities=probabilities,
        prediction_ids=prediction_ids,
        model_version=MODEL_VERSION,
    )


def run_server():
    """Run the API server."""
    uvicorn.run(
        "mlops_forge.api.api:app",
        host=settings.API.HOST,
        port=settings.API.PORT,
        reload=settings.API.RELOAD,
    )


# Helper functions for monitoring
def log_prediction(
    input_data: Dict[str, any],
    prediction: any,
    probability: Optional[float],
    prediction_id: str,
    latency: float,
):
    """Log prediction for monitoring.
    
    Args:
        input_data: Input features
        prediction: Prediction value
        probability: Prediction probability
        prediction_id: Unique identifier for the prediction
        latency: Prediction latency in seconds
    """
    logger.debug(
        f"Prediction logged: id={prediction_id}, prediction={prediction}, "
        f"probability={probability}, latency={latency:.4f}s"
    )
    # In a real application, you would log to a monitoring system
    # For example, using MLflow or a custom monitoring solution


def log_error(
    error: str,
    input_data: Dict[str, any],
    prediction_id: str,
):
    """Log error for monitoring.
    
    Args:
        error: Error message
        input_data: Input features
        prediction_id: Unique identifier for the prediction
    """
    logger.error(f"Prediction error logged: id={prediction_id}, error={error}")
    # In a real application, you would log to a monitoring system
