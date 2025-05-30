"""Metrics collection and monitoring for the MLOps Production System."""

import time
from functools import wraps
from typing import Callable, Dict, Any, List, Optional

from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import numpy as np
import pandas as pd

from mlops_production_system.config.settings import settings


# Initialize Prometheus metrics
prediction_counter = Counter(
    f"{settings.MONITORING.METRICS_PREFIX}_prediction_count",
    "Total number of predictions made"
)

prediction_error_counter = Counter(
    f"{settings.MONITORING.METRICS_PREFIX}_prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"]
)

prediction_latency = Histogram(
    f"{settings.MONITORING.METRICS_PREFIX}_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

feature_value_gauge = Gauge(
    f"{settings.MONITORING.METRICS_PREFIX}_feature_value",
    "Current feature values",
    ["feature_name"]
)

data_drift_gauge = Gauge(
    f"{settings.MONITORING.METRICS_PREFIX}_data_drift_score",
    "Data drift score (0-1)"
)

model_version_info = Gauge(
    f"{settings.MONITORING.METRICS_PREFIX}_model_version",
    "Current model version information",
    ["model_name", "version", "created_at"]
)

model_performance_gauge = Gauge(
    f"{settings.MONITORING.METRICS_PREFIX}_model_performance",
    "Current model performance metrics",
    ["metric_name"]
)

# Set common model metrics
model_accuracy_gauge = Gauge(
    f"{settings.MONITORING.METRICS_PREFIX}_model_accuracy",
    "Current model accuracy"
)

prediction_distribution = Histogram(
    f"{settings.MONITORING.METRICS_PREFIX}_prediction_distribution",
    "Distribution of prediction values",
    buckets=list(np.linspace(0, 1, 11))  # 0.0, 0.1, 0.2, ..., 1.0
)

batch_size_summary = Summary(
    f"{settings.MONITORING.METRICS_PREFIX}_batch_size",
    "Summary of batch prediction sizes"
)


def init_monitoring(port: Optional[int] = None) -> None:
    """Initialize the monitoring system.
    
    Args:
        port: Port to expose Prometheus metrics on. If None, uses the configured port.
    """
    monitoring_port = port or settings.MONITORING.PROMETHEUS_PORT
    start_http_server(monitoring_port)


def set_model_info(model_name: str, version: str, created_at: str) -> None:
    """Set the current model version information.
    
    Args:
        model_name: Name of the model
        version: Model version
        created_at: Timestamp when model was created
    """
    model_version_info.labels(
        model_name=model_name,
        version=version,
        created_at=created_at
    ).set(1)


def set_model_performance_metrics(metrics: Dict[str, float]) -> None:
    """Set the current model performance metrics.
    
    Args:
        metrics: Dictionary of metric names and values
    """
    for metric_name, value in metrics.items():
        model_performance_gauge.labels(metric_name=metric_name).set(value)
        
        # Also set specific gauges for common metrics
        if metric_name == "accuracy" or metric_name == "test_accuracy":
            model_accuracy_gauge.set(value)


def track_prediction_latency(func: Callable) -> Callable:
    """Decorator to track prediction latency.
    
    Args:
        func: Function to track
        
    Returns:
        Wrapped function with latency tracking
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            prediction_counter.inc()
            return result
        except Exception as e:
            prediction_error_counter.labels(
                error_type=type(e).__name__
            ).inc()
            raise
        finally:
            end_time = time.time()
            prediction_latency.observe(end_time - start_time)
    
    return wrapper


def track_feature_distribution(features: Dict[str, Any]) -> None:
    """Track the distribution of feature values.
    
    Args:
        features: Dictionary of feature names and values
    """
    for feature_name, value in features.items():
        # Only track numerical features
        if isinstance(value, (int, float)):
            feature_value_gauge.labels(feature_name=feature_name).set(value)


def track_prediction_distribution(predictions: List[float]) -> None:
    """Track the distribution of prediction values.
    
    Args:
        predictions: List of prediction values
    """
    for prediction in predictions:
        prediction_distribution.observe(prediction)
    
    batch_size_summary.observe(len(predictions))


def calculate_data_drift(
    reference_data: pd.DataFrame, 
    current_data: pd.DataFrame,
    features: List[str]
) -> float:
    """Calculate the data drift score between reference and current data.
    
    Args:
        reference_data: Reference data (training data)
        current_data: Current data (production data)
        features: List of features to consider
        
    Returns:
        Drift score between 0 and 1 (higher means more drift)
    """
    # Simple implementation using statistical distance
    drift_scores = []
    
    for feature in features:
        if feature in reference_data.columns and feature in current_data.columns:
            # Skip non-numeric features
            if not pd.api.types.is_numeric_dtype(reference_data[feature]) or \
               not pd.api.types.is_numeric_dtype(current_data[feature]):
                continue
                
            # Calculate mean and std for reference data
            ref_mean = reference_data[feature].mean()
            ref_std = reference_data[feature].std() or 1.0  # Avoid division by zero
            
            # Calculate mean for current data
            curr_mean = current_data[feature].mean()
            
            # Calculate standardized distance
            distance = abs(ref_mean - curr_mean) / ref_std
            
            # Normalize to 0-1 range using a simple sigmoid
            normalized_distance = 2 / (1 + np.exp(-distance)) - 1
            
            drift_scores.append(normalized_distance)
    
    # Average drift score
    if drift_scores:
        avg_drift = sum(drift_scores) / len(drift_scores)
        
        # Update the drift gauge
        data_drift_gauge.set(avg_drift)
        
        return avg_drift
    
    return 0.0


class DataDriftMonitor:
    """Monitor for detecting data drift in production."""
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        features: List[str],
        drift_threshold: float = 0.1
    ):
        """Initialize the data drift monitor.
        
        Args:
            reference_data: Reference data (training data)
            features: List of features to monitor
            drift_threshold: Threshold for significant drift (0-1)
        """
        self.reference_data = reference_data
        self.features = [f for f in features if f in reference_data.columns]
        self.drift_threshold = drift_threshold
        self.current_buffer = []
        self.buffer_size = 100  # Number of samples to collect before calculating drift
        
    def add_sample(self, sample: Dict[str, Any]) -> None:
        """Add a sample to the monitor.
        
        Args:
            sample: Dictionary of feature values
        """
        # Convert sample to appropriate format and add to buffer
        self.current_buffer.append(sample)
        
        # Calculate drift if buffer is full
        if len(self.current_buffer) >= self.buffer_size:
            self.calculate_drift()
            
    def calculate_drift(self) -> float:
        """Calculate current drift score based on collected samples.
        
        Returns:
            Current drift score
        """
        if not self.current_buffer:
            return 0.0
            
        # Convert buffer to DataFrame
        current_data = pd.DataFrame(self.current_buffer)
        
        # Calculate drift
        drift_score = calculate_data_drift(
            self.reference_data,
            current_data,
            self.features
        )
        
        # Clear buffer after calculation
        self.current_buffer = []
        
        return drift_score
        
    def is_drift_significant(self) -> bool:
        """Check if the current drift is significant.
        
        Returns:
            True if drift exceeds threshold, False otherwise
        """
        drift_score = data_drift_gauge._value.get()
        return drift_score > self.drift_threshold
