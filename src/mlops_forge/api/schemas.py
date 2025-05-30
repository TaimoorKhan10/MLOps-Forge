"""API schemas for data validation."""

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Schema for prediction input data."""
    
    features: Dict[str, Union[float, int, str]] = Field(
        ..., description="Dictionary of feature name to feature value"
    )
    
    class Config:
        """Pydantic configuration."""
        
        schema_extra = {
            "example": {
                "features": {
                    "feature_1": 0.5,
                    "feature_2": 10,
                    "feature_3": "category_a"
                }
            }
        }


class PredictionOutput(BaseModel):
    """Schema for prediction output data."""
    
    prediction: Union[float, int, str] = Field(
        ..., description="Prediction value"
    )
    probability: Optional[float] = Field(
        None, description="Prediction probability (for classification only)"
    )
    prediction_id: str = Field(
        ..., description="Unique identifier for the prediction"
    )
    model_version: str = Field(
        ..., description="Version of the model used for prediction"
    )
    
    class Config:
        """Pydantic configuration."""
        
        schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.95,
                "prediction_id": "123e4567-e89b-12d3-a456-426614174000",
                "model_version": "1.0.0"
            }
        }


class BatchPredictionInput(BaseModel):
    """Schema for batch prediction input data."""
    
    instances: List[Dict[str, Union[float, int, str]]] = Field(
        ..., description="List of feature dictionaries"
    )
    
    class Config:
        """Pydantic configuration."""
        
        schema_extra = {
            "example": {
                "instances": [
                    {
                        "feature_1": 0.5,
                        "feature_2": 10,
                        "feature_3": "category_a"
                    },
                    {
                        "feature_1": 0.7,
                        "feature_2": 20,
                        "feature_3": "category_b"
                    }
                ]
            }
        }


class BatchPredictionOutput(BaseModel):
    """Schema for batch prediction output data."""
    
    predictions: List[Union[float, int, str]] = Field(
        ..., description="List of prediction values"
    )
    probabilities: Optional[List[float]] = Field(
        None, description="List of prediction probabilities (for classification only)"
    )
    prediction_ids: List[str] = Field(
        ..., description="List of unique identifiers for each prediction"
    )
    model_version: str = Field(
        ..., description="Version of the model used for predictions"
    )
    
    class Config:
        """Pydantic configuration."""
        
        schema_extra = {
            "example": {
                "predictions": [1, 0],
                "probabilities": [0.95, 0.25],
                "prediction_ids": [
                    "123e4567-e89b-12d3-a456-426614174000",
                    "223e4567-e89b-12d3-a456-426614174001"
                ],
                "model_version": "1.0.0"
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response."""
    
    status: str = Field(..., description="Service status")
    model_version: str = Field(..., description="Model version")
    api_version: str = Field(..., description="API version")
    
    class Config:
        """Pydantic configuration."""
        
        schema_extra = {
            "example": {
                "status": "ok",
                "model_version": "1.0.0",
                "api_version": "v1"
            }
        }


class ModelMetadata(BaseModel):
    """Schema for model metadata."""
    
    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Version of the model")
    model_type: str = Field(..., description="Type of the model")
    created_at: str = Field(..., description="Timestamp when the model was created")
    features: List[str] = Field(..., description="List of features used by the model")
    metrics: Dict[str, float] = Field(
        ..., description="Dictionary of evaluation metrics"
    )
    
    class Config:
        """Pydantic configuration."""
        
        schema_extra = {
            "example": {
                "model_name": "random_forest_classifier",
                "model_version": "1.0.0",
                "model_type": "random_forest",
                "created_at": "2023-09-15T12:00:00Z",
                "features": ["feature_1", "feature_2", "feature_3"],
                "metrics": {
                    "accuracy": 0.95,
                    "f1": 0.94,
                    "precision": 0.93,
                    "recall": 0.92
                }
            }
        }
