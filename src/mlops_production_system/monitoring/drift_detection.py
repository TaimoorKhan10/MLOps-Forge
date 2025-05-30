"""Feature-level drift detection for ML models in production."""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
import mlflow
from prometheus_client import Gauge, Counter, Summary

from mlops_production_system.config.settings import settings
from mlops_production_system.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DriftStatus(Enum):
    """Status of drift detection."""
    
    NO_DRIFT = "no_drift"
    WARNING = "warning"
    DRIFT_DETECTED = "drift_detected"


class DriftMetricType(Enum):
    """Types of drift metrics."""
    
    KS_TEST = "ks_test"
    CHI2_TEST = "chi2_test"
    JS_DIVERGENCE = "js_divergence"
    PSI = "psi"
    WASSERSTEIN = "wasserstein"


# Prometheus metrics for drift detection
feature_drift_score = Gauge(
    f"{settings.MONITORING.METRICS_PREFIX}_feature_drift_score",
    "Drift score for individual features",
    ["feature_name", "metric_type"]
)

overall_drift_score = Gauge(
    f"{settings.MONITORING.METRICS_PREFIX}_overall_drift_score",
    "Overall drift score across all features"
)

drift_status = Gauge(
    f"{settings.MONITORING.METRICS_PREFIX}_drift_status",
    "Current drift status (0=no_drift, 1=warning, 2=drift_detected)"
)

drift_check_counter = Counter(
    f"{settings.MONITORING.METRICS_PREFIX}_drift_checks_total",
    "Total number of drift checks performed"
)

retraining_trigger_counter = Counter(
    f"{settings.MONITORING.METRICS_PREFIX}_retraining_triggers_total",
    "Total number of retraining triggers"
)

drift_check_latency = Summary(
    f"{settings.MONITORING.METRICS_PREFIX}_drift_check_latency_seconds",
    "Latency of drift detection checks"
)


class FeatureDriftDetector:
    """Detect data drift at the feature level."""
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        warning_threshold: float = 0.1,
        drift_threshold: float = 0.2,
        metrics: Optional[List[str]] = None
    ):
        """Initialize the drift detector.
        
        Args:
            reference_data: Reference data (training data)
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            warning_threshold: Threshold for drift warning
            drift_threshold: Threshold for drift detection
            metrics: List of drift metrics to use
        """
        self.reference_data = reference_data
        
        # Identify categorical and numerical features if not provided
        if categorical_features is None and numerical_features is None:
            self.categorical_features = reference_data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            
            self.numerical_features = reference_data.select_dtypes(
                include=["int", "float"]
            ).columns.tolist()
        else:
            self.categorical_features = categorical_features or []
            self.numerical_features = numerical_features or []
        
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        
        # Default metrics if none provided
        if metrics is None:
            metrics = [
                DriftMetricType.KS_TEST.value,
                DriftMetricType.CHI2_TEST.value,
                DriftMetricType.PSI.value
            ]
        
        self.metrics = [DriftMetricType(m) if isinstance(m, str) else m for m in metrics]
        
        # Store reference data statistics
        self.reference_stats = self._compute_reference_statistics()
        
        # Initialize drift results
        self.latest_drift_results = None
        self.drift_history = []
        
        logger.info(
            f"Initialized drift detector with {len(self.numerical_features)} numerical features "
            f"and {len(self.categorical_features)} categorical features"
        )
    
    def _compute_reference_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Compute statistics for reference data.
        
        Returns:
            Dictionary of feature statistics
        """
        stats = {}
        
        # Numerical features
        for feature in self.numerical_features:
            if feature not in self.reference_data.columns:
                continue
                
            feature_data = self.reference_data[feature].dropna()
            if len(feature_data) == 0:
                continue
                
            # Compute basic statistics
            stats[feature] = {
                "type": "numerical",
                "mean": feature_data.mean(),
                "std": feature_data.std(),
                "min": feature_data.min(),
                "max": feature_data.max(),
                "median": feature_data.median(),
                "histogram": np.histogram(feature_data, bins=10),
                "quantiles": feature_data.quantile([0.25, 0.5, 0.75]).to_dict()
            }
        
        # Categorical features
        for feature in self.categorical_features:
            if feature not in self.reference_data.columns:
                continue
                
            feature_data = self.reference_data[feature].dropna()
            if len(feature_data) == 0:
                continue
                
            # Compute value counts and frequencies
            value_counts = feature_data.value_counts()
            stats[feature] = {
                "type": "categorical",
                "value_counts": value_counts.to_dict(),
                "unique_count": len(value_counts),
                "frequencies": (value_counts / len(feature_data)).to_dict()
            }
        
        return stats
    
    def compute_drift(
        self,
        current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute drift between reference data and current data.
        
        Args:
            current_data: Current data to check for drift
            
        Returns:
            Dictionary with drift results
        """
        start_time = datetime.now()
        drift_check_counter.inc()
        
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "features": {},
            "overall_drift_score": 0.0,
            "drift_status": DriftStatus.NO_DRIFT.value,
            "metrics_used": [m.value for m in self.metrics],
            "num_samples": len(current_data)
        }
        
        feature_drift_scores = []
        
        # Check each numerical feature
        for feature in self.numerical_features:
            if feature not in self.reference_data.columns or feature not in current_data.columns:
                continue
                
            ref_data = self.reference_data[feature].dropna()
            cur_data = current_data[feature].dropna()
            
            if len(ref_data) == 0 or len(cur_data) == 0:
                continue
            
            feature_drift = {"type": "numerical", "metrics": {}}
            
            # Apply each drift metric
            for metric_type in self.metrics:
                if metric_type == DriftMetricType.KS_TEST:
                    # Kolmogorov-Smirnov test for numerical distributions
                    statistic, p_value = stats.ks_2samp(ref_data, cur_data)
                    drift_score = 1 - p_value  # Convert p-value to drift score (1 = high drift)
                    feature_drift["metrics"][metric_type.value] = {
                        "statistic": statistic,
                        "p_value": p_value,
                        "drift_score": drift_score
                    }
                    
                elif metric_type == DriftMetricType.PSI:
                    # Population Stability Index
                    psi_value = self._calculate_psi(ref_data, cur_data)
                    feature_drift["metrics"][metric_type.value] = {
                        "drift_score": psi_value
                    }
                    
                elif metric_type == DriftMetricType.WASSERSTEIN:
                    # Wasserstein distance (Earth Mover's Distance)
                    w_distance = stats.wasserstein_distance(ref_data, cur_data)
                    # Normalize by range
                    feature_range = self.reference_stats[feature]["max"] - self.reference_stats[feature]["min"]
                    if feature_range > 0:
                        normalized_distance = w_distance / feature_range
                    else:
                        normalized_distance = w_distance
                        
                    feature_drift["metrics"][metric_type.value] = {
                        "drift_score": normalized_distance
                    }
            
            # Calculate average drift score across metrics
            metric_scores = [m["drift_score"] for m in feature_drift["metrics"].values()]
            if metric_scores:
                avg_drift_score = sum(metric_scores) / len(metric_scores)
                feature_drift["drift_score"] = avg_drift_score
                feature_drift_scores.append(avg_drift_score)
                
                # Set drift status for this feature
                if avg_drift_score >= self.drift_threshold:
                    feature_drift["status"] = DriftStatus.DRIFT_DETECTED.value
                elif avg_drift_score >= self.warning_threshold:
                    feature_drift["status"] = DriftStatus.WARNING.value
                else:
                    feature_drift["status"] = DriftStatus.NO_DRIFT.value
                
                # Update Prometheus metrics
                feature_drift_score.labels(
                    feature_name=feature,
                    metric_type="average"
                ).set(avg_drift_score)
                
                for metric_type, metric_data in feature_drift["metrics"].items():
                    feature_drift_score.labels(
                        feature_name=feature,
                        metric_type=metric_type
                    ).set(metric_data["drift_score"])
            
            drift_results["features"][feature] = feature_drift
        
        # Check each categorical feature
        for feature in self.categorical_features:
            if feature not in self.reference_data.columns or feature not in current_data.columns:
                continue
                
            ref_data = self.reference_data[feature].dropna()
            cur_data = current_data[feature].dropna()
            
            if len(ref_data) == 0 or len(cur_data) == 0:
                continue
            
            feature_drift = {"type": "categorical", "metrics": {}}
            
            # Apply each drift metric
            for metric_type in self.metrics:
                if metric_type == DriftMetricType.CHI2_TEST:
                    # Chi-squared test for categorical distributions
                    chi2_result = self._calculate_chi2(feature, ref_data, cur_data)
                    if chi2_result is not None:
                        feature_drift["metrics"][metric_type.value] = {
                            "statistic": chi2_result["statistic"],
                            "p_value": chi2_result["p_value"],
                            "drift_score": chi2_result["drift_score"]
                        }
                
                elif metric_type == DriftMetricType.JS_DIVERGENCE:
                    # Jensen-Shannon divergence
                    js_div = self._calculate_js_divergence(feature, ref_data, cur_data)
                    if js_div is not None:
                        feature_drift["metrics"][metric_type.value] = {
                            "drift_score": js_div
                        }
                
                elif metric_type == DriftMetricType.PSI:
                    # PSI for categorical features
                    psi_value = self._calculate_categorical_psi(feature, ref_data, cur_data)
                    if psi_value is not None:
                        feature_drift["metrics"][metric_type.value] = {
                            "drift_score": psi_value
                        }
            
            # Calculate average drift score across metrics
            metric_scores = [m["drift_score"] for m in feature_drift["metrics"].values()]
            if metric_scores:
                avg_drift_score = sum(metric_scores) / len(metric_scores)
                feature_drift["drift_score"] = avg_drift_score
                feature_drift_scores.append(avg_drift_score)
                
                # Set drift status for this feature
                if avg_drift_score >= self.drift_threshold:
                    feature_drift["status"] = DriftStatus.DRIFT_DETECTED.value
                elif avg_drift_score >= self.warning_threshold:
                    feature_drift["status"] = DriftStatus.WARNING.value
                else:
                    feature_drift["status"] = DriftStatus.NO_DRIFT.value
                
                # Update Prometheus metrics
                feature_drift_score.labels(
                    feature_name=feature,
                    metric_type="average"
                ).set(avg_drift_score)
                
                for metric_type, metric_data in feature_drift["metrics"].items():
                    feature_drift_score.labels(
                        feature_name=feature,
                        metric_type=metric_type
                    ).set(metric_data["drift_score"])
            
            drift_results["features"][feature] = feature_drift
        
        # Calculate overall drift score as average of feature drift scores
        if feature_drift_scores:
            overall_score = sum(feature_drift_scores) / len(feature_drift_scores)
            drift_results["overall_drift_score"] = overall_score
            overall_drift_score.set(overall_score)
            
            # Set overall drift status
            if overall_score >= self.drift_threshold:
                drift_results["drift_status"] = DriftStatus.DRIFT_DETECTED.value
                drift_status.set(2)
            elif overall_score >= self.warning_threshold:
                drift_results["drift_status"] = DriftStatus.WARNING.value
                drift_status.set(1)
            else:
                drift_results["drift_status"] = DriftStatus.NO_DRIFT.value
                drift_status.set(0)
        
        # Record execution time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        drift_results["execution_time_seconds"] = execution_time
        drift_check_latency.observe(execution_time)
        
        # Store results
        self.latest_drift_results = drift_results
        self.drift_history.append(drift_results)
        
        return drift_results
    
    def _calculate_psi(
        self,
        reference_data: pd.Series,
        current_data: pd.Series,
        bins: int = 10,
        epsilon: float = 1e-6
    ) -> float:
        """Calculate Population Stability Index (PSI) for numerical data.
        
        Args:
            reference_data: Reference data distribution
            current_data: Current data distribution
            bins: Number of bins for histograms
            epsilon: Small value to avoid division by zero
            
        Returns:
            PSI value (higher means more drift)
        """
        # Create histograms
        ref_hist, bin_edges = np.histogram(reference_data, bins=bins)
        cur_hist, _ = np.histogram(current_data, bins=bin_edges)
        
        # Convert to frequencies
        ref_freq = ref_hist / (np.sum(ref_hist) + epsilon)
        cur_freq = cur_hist / (np.sum(cur_hist) + epsilon)
        
        # Add epsilon to avoid division by zero or log(0)
        ref_freq = np.array([max(x, epsilon) for x in ref_freq])
        cur_freq = np.array([max(x, epsilon) for x in cur_freq])
        
        # Calculate PSI
        psi_values = (cur_freq - ref_freq) * np.log(cur_freq / ref_freq)
        psi = np.sum(psi_values)
        
        return psi
    
    def _calculate_categorical_psi(
        self,
        feature: str,
        reference_data: pd.Series,
        current_data: pd.Series,
        epsilon: float = 1e-6
    ) -> Optional[float]:
        """Calculate PSI for categorical data.
        
        Args:
            feature: Feature name
            reference_data: Reference data distribution
            current_data: Current data distribution
            epsilon: Small value to avoid division by zero
            
        Returns:
            PSI value (higher means more drift)
        """
        if feature not in self.reference_stats:
            return None
        
        # Get reference frequencies
        ref_freqs = self.reference_stats[feature]["frequencies"]
        
        # Calculate current frequencies
        cur_value_counts = current_data.value_counts()
        cur_freqs = (cur_value_counts / len(current_data)).to_dict()
        
        # Make sure all categories are included
        all_categories = set(list(ref_freqs.keys()) + list(cur_freqs.keys()))
        
        # Calculate PSI
        psi = 0
        for category in all_categories:
            ref_freq = ref_freqs.get(category, epsilon)
            cur_freq = cur_freqs.get(category, epsilon)
            
            # Add epsilon to avoid division by zero or log(0)
            ref_freq = max(ref_freq, epsilon)
            cur_freq = max(cur_freq, epsilon)
            
            psi += (cur_freq - ref_freq) * np.log(cur_freq / ref_freq)
        
        return psi
    
    def _calculate_chi2(
        self,
        feature: str,
        reference_data: pd.Series,
        current_data: pd.Series
    ) -> Optional[Dict[str, float]]:
        """Calculate Chi-squared test for categorical data.
        
        Args:
            feature: Feature name
            reference_data: Reference data distribution
            current_data: Current data distribution
            
        Returns:
            Dictionary with test results
        """
        if feature not in self.reference_stats:
            return None
        
        # Get unique values
        unique_values = list(set(
            list(self.reference_stats[feature]["value_counts"].keys()) + 
            list(current_data.value_counts().keys())
        ))
        
        if len(unique_values) <= 1:
            return None
        
        # Calculate observed frequencies in current data
        cur_freqs = current_data.value_counts().to_dict()
        
        # Calculate expected frequencies based on reference data
        ref_freqs = self.reference_stats[feature]["frequencies"]
        expected_freqs = {
            val: ref_freqs.get(val, 0) * len(current_data) 
            for val in unique_values
        }
        
        # Prepare arrays for chi2 test
        observed = np.array([cur_freqs.get(val, 0) for val in unique_values])
        expected = np.array([expected_freqs.get(val, 0) for val in unique_values])
        
        # Remove categories with zero expected frequency
        mask = expected > 0
        if np.sum(mask) <= 1:
            return None
            
        observed = observed[mask]
        expected = expected[mask]
        
        # Calculate chi2 test
        statistic, p_value = stats.chisquare(observed, expected)
        
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_score": 1 - p_value  # Convert p-value to drift score
        }
    
    def _calculate_js_divergence(
        self,
        feature: str,
        reference_data: pd.Series,
        current_data: pd.Series,
        epsilon: float = 1e-6
    ) -> Optional[float]:
        """Calculate Jensen-Shannon divergence for categorical data.
        
        Args:
            feature: Feature name
            reference_data: Reference data distribution
            current_data: Current data distribution
            epsilon: Small value to avoid division by zero
            
        Returns:
            JS divergence (higher means more drift)
        """
        if feature not in self.reference_stats:
            return None
        
        # Get reference frequencies
        ref_freqs = self.reference_stats[feature]["frequencies"]
        
        # Calculate current frequencies
        cur_value_counts = current_data.value_counts()
        cur_freqs = (cur_value_counts / len(current_data)).to_dict()
        
        # Make sure all categories are included
        all_categories = set(list(ref_freqs.keys()) + list(cur_freqs.keys()))
        
        # Prepare probability distributions
        p = np.array([ref_freqs.get(cat, epsilon) for cat in all_categories])
        q = np.array([cur_freqs.get(cat, epsilon) for cat in all_categories])
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate midpoint distribution
        m = (p + q) / 2
        
        # Calculate KL divergences
        kl_pm = np.sum(p * np.log2(p / m))
        kl_qm = np.sum(q * np.log2(q / m))
        
        # Jensen-Shannon divergence is the average of KL divergences
        js_divergence = (kl_pm + kl_qm) / 2
        
        return js_divergence
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained based on drift detection.
        
        Returns:
            True if retraining is recommended
        """
        if self.latest_drift_results is None:
            return False
        
        # Recommend retraining if drift is detected
        if self.latest_drift_results["drift_status"] == DriftStatus.DRIFT_DETECTED.value:
            retraining_trigger_counter.inc()
            return True
        
        return False
    
    def save_results(self, filepath: str) -> None:
        """Save drift results to a file.
        
        Args:
            filepath: Path to save the results
        """
        if self.latest_drift_results is None:
            return
        
        with open(filepath, "w") as f:
            json.dump(self.latest_drift_results, f, indent=2)
        
        logger.info(f"Saved drift detection results to {filepath}")
    
    def log_to_mlflow(self, run_id: Optional[str] = None) -> None:
        """Log drift results to MLflow.
        
        Args:
            run_id: MLflow run ID (uses active run if None)
        """
        if self.latest_drift_results is None:
            return
        
        # Set up MLflow
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", settings.MLFLOW.TRACKING_URI)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Get active run or specified run
        if run_id is not None:
            with mlflow.start_run(run_id=run_id):
                self._log_drift_metrics()
        else:
            self._log_drift_metrics()
    
    def _log_drift_metrics(self) -> None:
        """Log drift metrics to MLflow."""
        # Log overall drift score
        mlflow.log_metric("overall_drift_score", self.latest_drift_results["overall_drift_score"])
        
        # Log feature-level drift scores
        for feature, drift_data in self.latest_drift_results["features"].items():
            if "drift_score" in drift_data:
                mlflow.log_metric(f"drift_score_{feature}", drift_data["drift_score"])
        
        # Log drift status
        mlflow.set_tag("drift_status", self.latest_drift_results["drift_status"])
        
        # Log as JSON artifact
        with open("drift_results.json", "w") as f:
            json.dump(self.latest_drift_results, f, indent=2)
        
        mlflow.log_artifact("drift_results.json")


class AutomaticRetrainingTrigger:
    """Trigger automatic model retraining based on drift detection."""
    
    def __init__(
        self,
        drift_detector: FeatureDriftDetector,
        retraining_frequency: int = 24,  # Hours
        min_samples_required: int = 1000,
        min_drift_score: float = 0.2,
        consecutive_triggers_required: int = 3
    ):
        """Initialize the retraining trigger.
        
        Args:
            drift_detector: Feature drift detector
            retraining_frequency: Minimum hours between retraining
            min_samples_required: Minimum number of samples required for retraining
            min_drift_score: Minimum drift score to consider retraining
            consecutive_triggers_required: Consecutive triggers required to retrain
        """
        self.drift_detector = drift_detector
        self.retraining_frequency = retraining_frequency
        self.min_samples_required = min_samples_required
        self.min_drift_score = min_drift_score
        self.consecutive_triggers_required = consecutive_triggers_required
        
        self.last_retrain_time = None
        self.trigger_count = 0
        self.collected_samples = 0
        
        logger.info(
            f"Initialized automatic retraining trigger (frequency={retraining_frequency}h, "
            f"min_samples={min_samples_required}, drift_threshold={min_drift_score})"
        )
    
    def add_samples(self, count: int) -> None:
        """Add samples to the counter.
        
        Args:
            count: Number of samples to add
        """
        self.collected_samples += count
    
    def check_drift(self, current_data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Check drift and determine if retraining is needed.
        
        Args:
            current_data: Current data to check for drift
            
        Returns:
            Tuple of (should_retrain, drift_results)
        """
        # Calculate drift
        drift_results = self.drift_detector.compute_drift(current_data)
        
        # Update sample count
        self.add_samples(len(current_data))
        
        # Check if retraining is needed
        should_retrain = self._should_retrain(drift_results)
        
        if should_retrain:
            logger.info(
                f"Automatic retraining triggered (drift_score={drift_results['overall_drift_score']:.4f}, "
                f"samples={self.collected_samples})"
            )
            
            # Reset after triggering retraining
            self.last_retrain_time = datetime.now()
            self.trigger_count = 0
            self.collected_samples = 0
        
        return should_retrain, drift_results
    
    def _should_retrain(self, drift_results: Dict[str, Any]) -> bool:
        """Determine if model should be retrained based on various criteria.
        
        Args:
            drift_results: Results from drift detection
            
        Returns:
            True if retraining is recommended
        """
        # Check if minimum samples threshold is met
        if self.collected_samples < self.min_samples_required:
            return False
        
        # Check retraining frequency
        if self.last_retrain_time is not None:
            hours_since_last_retrain = (
                datetime.now() - self.last_retrain_time
            ).total_seconds() / 3600
            
            if hours_since_last_retrain < self.retraining_frequency:
                return False
        
        # Check drift score
        if drift_results["overall_drift_score"] >= self.min_drift_score:
            self.trigger_count += 1
        else:
            self.trigger_count = 0
        
        # Check consecutive triggers
        return self.trigger_count >= self.consecutive_triggers_required
