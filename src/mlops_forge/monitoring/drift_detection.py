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

from mlops_forge.config.settings import settings
from mlops_forge.utils.logging_utils import get_logger

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
    PSI = "psi"
    JS_DIVERGENCE = "js_divergence"


# Prometheus metrics for drift detection
feature_drift_score = Gauge(
    f"{settings.MONITORING.METRICS_PREFIX}_feature_drift_score",
    "Drift score for each feature",
    ["feature", "metric"]
)

drift_status_counter = Counter(
    f"{settings.MONITORING.METRICS_PREFIX}_drift_status_total",
    "Count of drift status occurrences",
    ["status"]
)

drift_detection_duration = Summary(
    f"{settings.MONITORING.METRICS_PREFIX}_drift_detection_duration_seconds",
    "Time taken for drift detection"
)

retraining_trigger_counter = Counter(
    f"{settings.MONITORING.METRICS_PREFIX}_retraining_trigger_total",
    "Count of retraining triggers"
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
    
    def _compute_reference_statistics(self) -> Dict[str, Any]:
        """Compute statistics for reference data.
        
        Returns:
            Dictionary of feature statistics
        """
        stats = {}
        
        # Numerical features
        for feature in self.numerical_features:
            if feature in self.reference_data.columns:
                feature_data = self.reference_data[feature].dropna()
                stats[feature] = {
                    "mean": feature_data.mean(),
                    "std": feature_data.std(),
                    "min": feature_data.min(),
                    "max": feature_data.max(),
                    "median": feature_data.median(),
                    "histogram": np.histogram(feature_data, bins=10)
                }
        
        # Categorical features
        for feature in self.categorical_features:
            if feature in self.reference_data.columns:
                feature_data = self.reference_data[feature].dropna()
                stats[feature] = {
                    "value_counts": feature_data.value_counts(normalize=True).to_dict()
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
        with drift_detection_duration.time():
            drift_results = {
                "timestamp": datetime.now().isoformat(),
                "features": {},
                "overall": {
                    "status": DriftStatus.NO_DRIFT.value,
                    "drift_score": 0.0,
                    "drifted_features_count": 0,
                    "warning_features_count": 0
                }
            }
            
            drifted_features = 0
            warning_features = 0
            total_drift_score = 0.0
            feature_count = 0
            
            # Numerical features
            for feature in self.numerical_features:
                if feature not in current_data.columns or feature not in self.reference_data.columns:
                    logger.warning(f"Feature {feature} not found in data")
                    continue
                
                ref_data = self.reference_data[feature].dropna()
                cur_data = current_data[feature].dropna()
                
                if len(cur_data) < 10:
                    logger.warning(f"Not enough data for feature {feature}, skipping drift detection")
                    continue
                
                feature_count += 1
                feature_results = {"type": "numerical", "metrics": {}}
                max_score = 0.0
                
                # KS test for numerical features
                if DriftMetricType.KS_TEST in self.metrics:
                    ks_stat, p_value = stats.ks_2samp(ref_data, cur_data)
                    score = 1.0 - p_value  # Higher score means more drift
                    
                    feature_results["metrics"]["ks_test"] = {
                        "statistic": float(ks_stat),
                        "p_value": float(p_value),
                        "score": float(score)
                    }
                    
                    feature_drift_score.labels(feature=feature, metric="ks_test").set(score)
                    max_score = max(max_score, score)
                
                # PSI for numerical features
                if DriftMetricType.PSI in self.metrics:
                    psi_value = self._calculate_psi(ref_data, cur_data)
                    score = min(psi_value, 1.0)  # Normalize to 0-1
                    
                    feature_results["metrics"]["psi"] = {
                        "value": float(psi_value),
                        "score": float(score)
                    }
                    
                    feature_drift_score.labels(feature=feature, metric="psi").set(score)
                    max_score = max(max_score, score)
                
                # Determine status based on highest score
                if max_score >= self.drift_threshold:
                    status = DriftStatus.DRIFT_DETECTED.value
                    drifted_features += 1
                elif max_score >= self.warning_threshold:
                    status = DriftStatus.WARNING.value
                    warning_features += 1
                else:
                    status = DriftStatus.NO_DRIFT.value
                
                feature_results["status"] = status
                feature_results["drift_score"] = float(max_score)
                drift_results["features"][feature] = feature_results
                total_drift_score += max_score
            
            # Categorical features
            for feature in self.categorical_features:
                if feature not in current_data.columns or feature not in self.reference_data.columns:
                    logger.warning(f"Feature {feature} not found in data")
                    continue
                
                ref_data = self.reference_data[feature].dropna()
                cur_data = current_data[feature].dropna()
                
                if len(cur_data) < 10:
                    logger.warning(f"Not enough data for feature {feature}, skipping drift detection")
                    continue
                
                feature_count += 1
                feature_results = {"type": "categorical", "metrics": {}}
                max_score = 0.0
                
                # Chi-squared test for categorical features
                if DriftMetricType.CHI2_TEST in self.metrics:
                    chi2_results = self._calculate_chi2(feature, ref_data, cur_data)
                    score = 1.0 - chi2_results.get("p_value", 0.0)  # Higher score means more drift
                    
                    feature_results["metrics"]["chi2_test"] = {
                        "statistic": chi2_results.get("statistic", 0.0),
                        "p_value": chi2_results.get("p_value", 1.0),
                        "score": float(score)
                    }
                    
                    feature_drift_score.labels(feature=feature, metric="chi2_test").set(score)
                    max_score = max(max_score, score)
                
                # PSI for categorical features
                if DriftMetricType.PSI in self.metrics:
                    psi_value = self._calculate_categorical_psi(feature, ref_data, cur_data)
                    score = min(psi_value, 1.0)  # Normalize to 0-1
                    
                    feature_results["metrics"]["psi"] = {
                        "value": float(psi_value),
                        "score": float(score)
                    }
                    
                    feature_drift_score.labels(feature=feature, metric="psi").set(score)
                    max_score = max(max_score, score)
                
                # JS divergence for categorical features
                if DriftMetricType.JS_DIVERGENCE in self.metrics:
                    js_value = self._calculate_js_divergence(feature, ref_data, cur_data)
                    score = float(js_value)  # Already 0-1
                    
                    feature_results["metrics"]["js_divergence"] = {
                        "value": float(js_value),
                        "score": float(score)
                    }
                    
                    feature_drift_score.labels(feature=feature, metric="js_divergence").set(score)
                    max_score = max(max_score, score)
                
                # Determine status based on highest score
                if max_score >= self.drift_threshold:
                    status = DriftStatus.DRIFT_DETECTED.value
                    drifted_features += 1
                elif max_score >= self.warning_threshold:
                    status = DriftStatus.WARNING.value
                    warning_features += 1
                else:
                    status = DriftStatus.NO_DRIFT.value
                
                feature_results["status"] = status
                feature_results["drift_score"] = float(max_score)
                drift_results["features"][feature] = feature_results
                total_drift_score += max_score
            
            # Calculate overall drift score and status
            if feature_count > 0:
                avg_drift_score = total_drift_score / feature_count
                drift_results["overall"]["drift_score"] = float(avg_drift_score)
                drift_results["overall"]["drifted_features_count"] = drifted_features
                drift_results["overall"]["warning_features_count"] = warning_features
                
                # Overall status is based on proportion of drifted features
                drift_proportion = drifted_features / feature_count
                warning_proportion = warning_features / feature_count
                
                if drift_proportion >= 0.1:  # 10% or more features have significant drift
                    overall_status = DriftStatus.DRIFT_DETECTED.value
                elif warning_proportion >= 0.2:  # 20% or more features have warning level drift
                    overall_status = DriftStatus.WARNING.value
                else:
                    overall_status = DriftStatus.NO_DRIFT.value
                
                drift_results["overall"]["status"] = overall_status
                drift_status_counter.labels(status=overall_status).inc()
            
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
        try:
            # Create bins based on reference data
            bin_edges = np.histogram_bin_edges(reference_data, bins=bins)
            
            # Calculate histograms
            ref_counts, _ = np.histogram(reference_data, bins=bin_edges)
            cur_counts, _ = np.histogram(current_data, bins=bin_edges)
            
            # Convert to proportions
            ref_props = ref_counts / (ref_counts.sum() + epsilon)
            cur_props = cur_counts / (cur_counts.sum() + epsilon)
            
            # Add epsilon to avoid division by zero
            ref_props = np.clip(ref_props, epsilon, None)
            cur_props = np.clip(cur_props, epsilon, None)
            
            # Calculate PSI
            psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
            
            return float(psi)
        except Exception as e:
            logger.error(f"Error calculating PSI: {str(e)}")
            return 0.0
    
    def _calculate_categorical_psi(
            self,
            feature: str,
            reference_data: pd.Series,
            current_data: pd.Series,
            epsilon: float = 1e-6
        ) -> float:
        """Calculate PSI for categorical data.
        
        Args:
            feature: Feature name
            reference_data: Reference data distribution
            current_data: Current data distribution
            epsilon: Small value to avoid division by zero
            
        Returns:
            PSI value (higher means more drift)
        """
        try:
            # Get value counts for each category
            ref_counts = reference_data.value_counts(normalize=True)
            cur_counts = current_data.value_counts(normalize=True)
            
            # Get all unique categories
            all_categories = set(ref_counts.index).union(set(cur_counts.index))
            
            # Initialize PSI
            psi = 0.0
            
            # Calculate PSI for each category
            for category in all_categories:
                ref_prop = ref_counts.get(category, 0.0) + epsilon
                cur_prop = cur_counts.get(category, 0.0) + epsilon
                
                psi += (cur_prop - ref_prop) * np.log(cur_prop / ref_prop)
            
            return float(psi)
        except Exception as e:
            logger.error(f"Error calculating categorical PSI for {feature}: {str(e)}")
            return 0.0
    
    def _calculate_chi2(
            self,
            feature: str,
            reference_data: pd.Series,
            current_data: pd.Series
        ) -> Dict[str, float]:
        """Calculate Chi-squared test for categorical data.
        
        Args:
            feature: Feature name
            reference_data: Reference data distribution
            current_data: Current data distribution
            
        Returns:
            Dictionary with test results
        """
        try:
            # Get all unique categories
            all_categories = set(reference_data.unique()).union(set(current_data.unique()))
            
            # Create contingency table
            ref_counts = pd.Series(0, index=all_categories)
            cur_counts = pd.Series(0, index=all_categories)
            
            # Fill in actual counts
            ref_counts.update(reference_data.value_counts())
            cur_counts.update(current_data.value_counts())
            
            # Contingency table as [ref_counts, cur_counts]
            contingency = np.vstack([ref_counts.values, cur_counts.values])
            
            # Calculate chi-squared test
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            
            return {
                "statistic": float(chi2),
                "p_value": float(p),
                "dof": int(dof)
            }
        except Exception as e:
            logger.error(f"Error calculating Chi-squared test for {feature}: {str(e)}")
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "dof": 0
            }
    
    def _calculate_js_divergence(
            self,
            feature: str,
            reference_data: pd.Series,
            current_data: pd.Series,
            epsilon: float = 1e-6
        ) -> float:
        """Calculate Jensen-Shannon divergence for categorical data.
        
        Args:
            feature: Feature name
            reference_data: Reference data distribution
            current_data: Current data distribution
            epsilon: Small value to avoid division by zero
            
        Returns:
            JS divergence (higher means more drift)
        """
        try:
            # Get value counts for each category
            ref_counts = reference_data.value_counts(normalize=True)
            cur_counts = current_data.value_counts(normalize=True)
            
            # Get all unique categories
            all_categories = set(ref_counts.index).union(set(cur_counts.index))
            
            # Create probability distributions
            p = np.zeros(len(all_categories))
            q = np.zeros(len(all_categories))
            
            # Fill in probabilities
            for i, category in enumerate(all_categories):
                p[i] = ref_counts.get(category, 0.0) + epsilon
                q[i] = cur_counts.get(category, 0.0) + epsilon
            
            # Normalize
            p = p / p.sum()
            q = q / q.sum()
            
            # Compute JS divergence
            m = 0.5 * (p + q)
            js_div = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))
            
            return float(js_div)
        except Exception as e:
            logger.error(f"Error calculating JS divergence for {feature}: {str(e)}")
            return 0.0
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained based on drift detection.
        
        Returns:
            True if retraining is recommended
        """
        if self.latest_drift_results is None:
            return False
        
        # Check overall drift status
        overall_status = self.latest_drift_results["overall"]["status"]
        if overall_status == DriftStatus.DRIFT_DETECTED.value:
            # Check proportion of drifted features
            feature_count = len(self.numerical_features) + len(self.categorical_features)
            drifted_count = self.latest_drift_results["overall"]["drifted_features_count"]
            
            if feature_count > 0 and drifted_count / feature_count >= 0.2:
                # At least 20% of features have drifted
                logger.warning(f"Retraining recommended: {drifted_count}/{feature_count} features have significant drift")
                return True
        
        return False
    
    def save_results(self, filepath: str):
        """Save drift results to a file.
        
        Args:
            filepath: Path to save the results
        """
        if self.latest_drift_results is None:
            logger.warning("No drift results to save")
            return
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.latest_drift_results, f, indent=2)
            logger.info(f"Drift results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving drift results: {str(e)}")
    
    def log_to_mlflow(self, run_id: Optional[str] = None):
        """Log drift results to MLflow.
        
        Args:
            run_id: MLflow run ID (uses active run if None)
        """
        if self.latest_drift_results is None:
            logger.warning("No drift results to log to MLflow")
            return
        
        try:
            # Get active run or create a new one
            active_run = mlflow.active_run()
            if active_run is None and run_id is None:
                active_run = mlflow.start_run()
            elif run_id is not None:
                active_run = mlflow.start_run(run_id=run_id)
            
            # Log overall drift metrics
            self._log_drift_metrics()
            
            # Save and log drift results as JSON artifact
            temp_path = f"drift_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.save_results(temp_path)
            mlflow.log_artifact(temp_path)
            
            # Clean up temp file
            os.remove(temp_path)
            
            logger.info(f"Drift results logged to MLflow run {active_run.info.run_id}")
        except Exception as e:
            logger.error(f"Error logging drift results to MLflow: {str(e)}")
    
    def _log_drift_metrics(self):
        """Log drift metrics to MLflow."""
        try:
            # Log overall drift score
            overall_score = self.latest_drift_results["overall"]["drift_score"]
            mlflow.log_metric("drift_score_overall", overall_score)
            
            # Log feature-level drift scores
            for feature, results in self.latest_drift_results["features"].items():
                feature_score = results["drift_score"]
                mlflow.log_metric(f"drift_score_{feature}", feature_score)
            
            # Log counts of drifted and warning features
            drifted_count = self.latest_drift_results["overall"]["drifted_features_count"]
            warning_count = self.latest_drift_results["overall"]["warning_features_count"]
            mlflow.log_metrics({
                "drifted_features_count": drifted_count,
                "warning_features_count": warning_count
            })
        except Exception as e:
            logger.error(f"Error logging drift metrics to MLflow: {str(e)}")


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
    
    def add_samples(self, count: int):
        """Add samples to the counter.
        
        Args:
            count: Number of samples to add
        """
        self.collected_samples += count
        logger.debug(f"Added {count} samples, total collected: {self.collected_samples}")
    
    def check_drift(self, current_data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Check drift and determine if retraining is needed.
        
        Args:
            current_data: Current data to check for drift
            
        Returns:
            Tuple of (should_retrain, drift_results)
        """
        # Compute drift
        drift_results = self.drift_detector.compute_drift(current_data)
        
        # Check if retraining is needed
        should_retrain = self._should_retrain(drift_results)
        
        if should_retrain:
            retraining_trigger_counter.inc()
            logger.info("Automatic retraining triggered")
            
            # Reset counters after triggering
            self.trigger_count = 0
            self.collected_samples = 0
            self.last_retrain_time = datetime.now()
        
        return should_retrain, drift_results
    
    def _should_retrain(self, drift_results: Dict[str, Any]) -> bool:
        """Determine if model should be retrained based on various criteria.
        
        Args:
            drift_results: Results from drift detection
            
        Returns:
            True if retraining is recommended
        """
        # Check if we have enough samples
        if self.collected_samples < self.min_samples_required:
            logger.debug(
                f"Not enough samples for retraining: {self.collected_samples}/{self.min_samples_required}"
            )
            return False
        
        # Check if enough time has passed since last retraining
        if self.last_retrain_time is not None:
            hours_since_last = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
            if hours_since_last < self.retraining_frequency:
                logger.debug(
                    f"Not enough time since last retraining: {hours_since_last:.1f}/{self.retraining_frequency}h"
                )
                return False
        
        # Check if drift score is high enough
        overall_drift_score = drift_results["overall"]["drift_score"]
        if overall_drift_score < self.min_drift_score:
            logger.debug(
                f"Drift score too low for retraining: {overall_drift_score:.3f}/{self.min_drift_score}"
            )
            self.trigger_count = 0  # Reset counter if drift score is low
            return False
        
        # Increment trigger counter
        self.trigger_count += 1
        logger.debug(f"Drift trigger count: {self.trigger_count}/{self.consecutive_triggers_required}")
        
        # Check if we have enough consecutive triggers
        if self.trigger_count >= self.consecutive_triggers_required:
            return True
        
        return False
