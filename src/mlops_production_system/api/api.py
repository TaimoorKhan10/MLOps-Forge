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

from mlops_production_system.api.schemas import (
    PredictionInput,
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
    HealthResponse,
    ModelMetadata,
)
from mlops_production_system.config.settings import settings
from mlops_production_system.models.model_trainer import get_model_trainer
from mlops_production_system.monitoring.monitoring import log_prediction, log_error
from mlops_production_system.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for serving ML predictions",
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

# Prometheus metrics
PREDICTION_COUNT = Counter(
    "prediction_count", "Number of predictions", ["model_version", "status"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency in seconds", ["model_version"]
)

# Load model on startup
model_trainer = None
model_metadata = None


@app.on_event("startup")
async def startup_event():
    """Load model and other resources on startup."""
    global model_trainer, model_metadata
    
    logger.info("Loading model...")
    
    try:
        # Create model trainer
        model_trainer = get_model_trainer(
            model_type=settings.MODEL.MODEL_NAME,
            experiment_name=settings.MLFLOW.EXPERIMENT_NAME,
        )
        
        # Load model from registry
        model_path = f"{settings.MODEL.MODEL_DIR}/{settings.MODEL.MODEL_NAME}_model.joblib"
        model_trainer.load_model(model_path)
        
        # Load model metadata
        import json
        from pathlib import Path
        
        metadata_path = Path(settings.MODEL.MODEL_DIR) / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                model_metadata = json.load(f)
        else:
            # Create simple metadata if file doesn't exist
            model_metadata = {
                "model_name": settings.MODEL.MODEL_NAME,
                "model_version": settings.MODEL.MODEL_VERSION,
                "model_type": settings.MODEL.MODEL_NAME,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "features": [],
                "metrics": {},
            }
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Release resources on shutdown."""
    logger.info("Shutting down API")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    if model_trainer is None or model_trainer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "ok",
        "model_version": settings.MODEL.MODEL_VERSION,
        "api_version": "v1",
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metadata", response_model=ModelMetadata, tags=["Model"])
async def get_model_metadata():
    """Get model metadata."""
    if model_metadata is None:
        raise HTTPException(status_code=404, detail="Model metadata not found")
    
    return model_metadata


@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict(
    data: PredictionInput, background_tasks: BackgroundTasks
):
    """Make a single prediction."""
    if model_trainer is None or model_trainer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Convert input to DataFrame
        features_df = pd.DataFrame([data.features])
        
        # Make prediction
        prediction = model_trainer.predict(features_df)[0]
        
        # Make probability prediction if possible
        probability = None
        try:
            probability = float(model_trainer.predict_proba(features_df)[0][1])
        except (AttributeError, IndexError):
            logger.warning("Model does not support predict_proba")
        
        # Generate prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Create response
        response = {
            "prediction": prediction,
            "probability": probability,
            "prediction_id": prediction_id,
            "model_version": settings.MODEL.MODEL_VERSION,
        }
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Update metrics
        PREDICTION_COUNT.labels(
            model_version=settings.MODEL.MODEL_VERSION, status="success"
        ).inc()
        PREDICTION_LATENCY.labels(
            model_version=settings.MODEL.MODEL_VERSION
        ).observe(latency)
        
        # Log prediction asynchronously
        background_tasks.add_task(
            log_prediction,
            prediction_id=prediction_id,
            features=data.features,
            prediction=prediction,
            probability=probability,
            latency=latency,
            model_version=settings.MODEL.MODEL_VERSION,
        )
        
        return response
        
    except Exception as e:
        # Update metrics
        PREDICTION_COUNT.labels(
            model_version=settings.MODEL.MODEL_VERSION, status="error"
        ).inc()
        
        # Log error asynchronously
        background_tasks.add_task(
            log_error,
            error=str(e),
            endpoint="/predict",
            payload=data.dict(),
        )
        
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.post("/batch-predict", response_model=BatchPredictionOutput, tags=["Predictions"])
async def batch_predict(
    data: BatchPredictionInput, background_tasks: BackgroundTasks
):
    """Make batch predictions."""
    if model_trainer is None or model_trainer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Convert input to DataFrame
        features_df = pd.DataFrame(data.instances)
        
        # Make predictions
        predictions = model_trainer.predict(features_df).tolist()
        
        # Make probability predictions if possible
        probabilities = None
        try:
            probabilities = model_trainer.predict_proba(features_df)[:, 1].tolist()
        except (AttributeError, IndexError):
            logger.warning("Model does not support predict_proba")
        
        # Generate prediction IDs
        prediction_ids = [str(uuid.uuid4()) for _ in range(len(predictions))]
        
        # Create response
        response = {
            "predictions": predictions,
            "probabilities": probabilities,
            "prediction_ids": prediction_ids,
            "model_version": settings.MODEL.MODEL_VERSION,
        }
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Update metrics
        PREDICTION_COUNT.labels(
            model_version=settings.MODEL.MODEL_VERSION, status="success"
        ).inc(len(predictions))
        PREDICTION_LATENCY.labels(
            model_version=settings.MODEL.MODEL_VERSION
        ).observe(latency)
        
        # Log predictions asynchronously
        for i, prediction_id in enumerate(prediction_ids):
            background_tasks.add_task(
                log_prediction,
                prediction_id=prediction_id,
                features=data.instances[i],
                prediction=predictions[i],
                probability=probabilities[i] if probabilities else None,
                latency=latency / len(predictions),
                model_version=settings.MODEL.MODEL_VERSION,
            )
        
        return response
        
    except Exception as e:
        # Update metrics
        PREDICTION_COUNT.labels(
            model_version=settings.MODEL.MODEL_VERSION, status="error"
        ).inc(len(data.instances))
        
        # Log error asynchronously
        background_tasks.add_task(
            log_error,
            error=str(e),
            endpoint="/batch-predict",
            payload=data.dict(),
        )
        
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {e}")


def run_server():
    """Run the API server."""
    uvicorn.run(
        "mlops_production_system.api.api:app",
        host=settings.API.HOST,
        port=settings.API.PORT,
        reload=settings.API.RELOAD,
        workers=settings.API.WORKERS,
        log_level="info",
    )
