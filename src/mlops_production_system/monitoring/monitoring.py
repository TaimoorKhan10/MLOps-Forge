"""Monitoring module for tracking model performance and data drift."""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab, NumTargetDriftTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from mlops_production_system.config.settings import settings
from mlops_production_system.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Prometheus metrics
PREDICTION_DRIFT_SCORE = Gauge(
    "prediction_drift_score", "Data drift score for model predictions", ["model_version"]
)
FEATURE_DRIFT_SCORE = Gauge(
    "feature_drift_score", "Data drift score for features", ["feature", "model_version"]
)
ERROR_COUNT = Counter(
    "error_count", "Number of errors", ["endpoint", "model_version"]
)
PREDICTION_LATENCY_GAUGE = Gauge(
    "prediction_latency_gauge", "Average prediction latency in seconds", ["model_version"]
)


class ModelMonitor:
    """Model monitoring class."""

    def __init__(
        self,
        reference_data_path: Optional[str] = None,
        monitoring_data_path: Optional[str] = None,
    ):
        """Initialize model monitor.
        
        Args:
            reference_data_path: Path to reference data for drift detection.
            monitoring_data_path: Path to save monitoring data.
        """
        self.reference_data_path = reference_data_path
        self.monitoring_data_path = monitoring_data_path or os.path.join(
            settings.DATA.PROCESSED_DATA_DIR, "monitoring"
        )
        
        # Create monitoring directory if it doesn't exist
        os.makedirs(self.monitoring_data_path, exist_ok=True)
        
        # Load reference data if provided
        self.reference_data = None
        if self.reference_data_path and os.path.exists(self.reference_data_path):
            self.load_reference_data()
            
        # Initialize monitoring data
        self.current_data = []
        
    def load_reference_data(self):
        """Load reference data for drift detection."""
        if not os.path.exists(self.reference_data_path):
            logger.warning(f"Reference data path {self.reference_data_path} does not exist")
            return
            
        # Load reference data
        try:
            if self.reference_data_path.endswith(".csv"):
                self.reference_data = pd.read_csv(self.reference_data_path)
            elif self.reference_data_path.endswith(".parquet"):
                self.reference_data = pd.read_parquet(self.reference_data_path)
            else:
                logger.warning(f"Unsupported reference data format: {self.reference_data_path}")
                return
                
            logger.info(f"Loaded reference data with shape {self.reference_data.shape}")
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            
    def add_monitoring_data(self, data: Dict[str, Any]):
        """Add data for monitoring.
        
        Args:
            data: Dictionary of data to add.
        """
        # Add timestamp
        data["timestamp"] = datetime.now().isoformat()
        
        # Add to current data
        self.current_data.append(data)
        
        # Save data periodically
        if len(self.current_data) >= 100:
            self.save_monitoring_data()
            
    def save_monitoring_data(self):
        """Save monitoring data to file."""
        if not self.current_data:
            logger.info("No monitoring data to save")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(self.current_data)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"monitoring_data_{timestamp}.csv"
        filepath = os.path.join(self.monitoring_data_path, filename)
        
        # Save to file
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(self.current_data)} monitoring records to {filepath}")
        
        # Clear current data
        self.current_data = []
        
    def detect_drift(self, current_data: pd.DataFrame = None) -> Dict[str, float]:
        """Detect data drift.
        
        Args:
            current_data: Current data to compare with reference data.
                If None, uses the collected monitoring data.
        
        Returns:
            Dictionary of drift scores.
        """
        if self.reference_data is None:
            logger.warning("Reference data not loaded, cannot detect drift")
            return {}
            
        # Use provided data or collected monitoring data
        if current_data is None:
            if not self.current_data:
                logger.warning("No monitoring data available for drift detection")
                return {}
                
            current_data = pd.DataFrame(self.current_data)
            
        # Select only common columns
        common_columns = list(set(self.reference_data.columns) & set(current_data.columns))
        if not common_columns:
            logger.warning("No common columns between reference and current data")
            return {}
            
        reference_data = self.reference_data[common_columns]
        current_data = current_data[common_columns]
        
        try:
            # Create data drift profile
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])
            data_drift_profile.calculate(reference_data, current_data)
            
            # Extract drift scores
            profile_dict = json.loads(data_drift_profile.json())
            drift_scores = {}
            
            # Overall drift score
            drift_scores["overall"] = profile_dict["data_drift"]["data"]["metrics"]["dataset_drift"]
            
            # Feature drift scores
            for feature in profile_dict["data_drift"]["data"]["metrics"]["feature_drift"]:
                feature_name = feature["feature_name"]
                drift_scores[feature_name] = feature["drift_score"]
                
            # Update Prometheus metrics
            PREDICTION_DRIFT_SCORE.labels(
                model_version=settings.MODEL.MODEL_VERSION
            ).set(drift_scores["overall"])
            
            for feature, score in drift_scores.items():
                if feature != "overall":
                    FEATURE_DRIFT_SCORE.labels(
                        feature=feature, model_version=settings.MODEL.MODEL_VERSION
                    ).set(score)
                    
            logger.info(f"Drift detection completed: overall_score={drift_scores['overall']}")
            
            return drift_scores
            
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return {}
            
    def generate_drift_dashboard(
        self, output_path: Optional[str] = None, current_data: pd.DataFrame = None
    ) -> str:
        """Generate data drift dashboard.
        
        Args:
            output_path: Path to save the dashboard.
            current_data: Current data to compare with reference data.
                If None, uses the collected monitoring data.
        
        Returns:
            Path to the generated dashboard.
        """
        if self.reference_data is None:
            logger.warning("Reference data not loaded, cannot generate dashboard")
            return ""
            
        # Use provided data or collected monitoring data
        if current_data is None:
            if not self.current_data:
                logger.warning("No monitoring data available for dashboard generation")
                return ""
                
            current_data = pd.DataFrame(self.current_data)
            
        # Determine output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.monitoring_data_path, f"drift_dashboard_{timestamp}.html"
            )
            
        try:
            # Create dashboard with data drift tab
            dashboard = Dashboard(tabs=[DataDriftTab()])
            dashboard.calculate(self.reference_data, current_data)
            
            # Save dashboard
            dashboard.save(output_path)
            logger.info(f"Drift dashboard saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating drift dashboard: {e}")
            return ""
            
    def start_prometheus_server(self, port: Optional[int] = None):
        """Start Prometheus metrics server.
        
        Args:
            port: Port to run the server on.
        """
        port = port or settings.MONITORING.PROMETHEUS_PORT
        
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Error starting Prometheus server: {e}")


# Global model monitor instance
model_monitor = ModelMonitor()


def log_prediction(
    prediction_id: str,
    features: Dict[str, Any],
    prediction: Any,
    probability: Optional[float] = None,
    latency: Optional[float] = None,
    model_version: Optional[str] = None,
):
    """Log prediction for monitoring.
    
    Args:
        prediction_id: Unique identifier for the prediction.
        features: Input features.
        prediction: Prediction value.
        probability: Prediction probability.
        latency: Prediction latency in seconds.
        model_version: Model version.
    """
    # Create monitoring data
    monitoring_data = {
        "prediction_id": prediction_id,
        "prediction": prediction,
        "probability": probability,
        "latency": latency,
        "model_version": model_version or settings.MODEL.MODEL_VERSION,
    }
    
    # Add features
    monitoring_data.update(features)
    
    # Add to model monitor
    model_monitor.add_monitoring_data(monitoring_data)
    
    # Update latency metric if provided
    if latency is not None:
        PREDICTION_LATENCY_GAUGE.labels(
            model_version=model_version or settings.MODEL.MODEL_VERSION
        ).set(latency)


def log_error(
    error: str,
    endpoint: str,
    payload: Dict[str, Any],
    model_version: Optional[str] = None,
):
    """Log error for monitoring.
    
    Args:
        error: Error message.
        endpoint: API endpoint where the error occurred.
        payload: Input payload.
        model_version: Model version.
    """
    # Create monitoring data
    monitoring_data = {
        "error": error,
        "endpoint": endpoint,
        "model_version": model_version or settings.MODEL.MODEL_VERSION,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Add to model monitor
    model_monitor.add_monitoring_data(monitoring_data)
    
    # Update error count metric
    ERROR_COUNT.labels(
        endpoint=endpoint, model_version=model_version or settings.MODEL.MODEL_VERSION
    ).inc()
    
    # Log error
    logger.error(f"Error on {endpoint}: {error}")
