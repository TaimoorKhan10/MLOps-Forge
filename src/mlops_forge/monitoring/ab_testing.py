"""A/B testing framework for ML models in production."""

import random
import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from enum import Enum
from datetime import datetime
import threading

import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge

from mlops_forge.config.settings import settings
from mlops_forge.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ExperimentStatus(Enum):
    """Status of an A/B test experiment."""
    
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"


class AllocationStrategy(Enum):
    """Traffic allocation strategy for A/B testing."""
    
    RANDOM = "random"
    STICKY = "sticky"
    TIME_BASED = "time_based"
    FEATURE_BASED = "feature_based"


# Prometheus metrics for A/B testing
ab_test_requests = Counter(
    f"{settings.MONITORING.METRICS_PREFIX}_ab_test_requests_total",
    "Total number of A/B test requests",
    ["experiment_id", "variant"]
)

ab_test_errors = Counter(
    f"{settings.MONITORING.METRICS_PREFIX}_ab_test_errors_total",
    "Total number of A/B test errors",
    ["experiment_id", "variant", "error_type"]
)

ab_test_latency = Histogram(
    f"{settings.MONITORING.METRICS_PREFIX}_ab_test_latency_seconds",
    "A/B test request latency in seconds",
    ["experiment_id", "variant"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]
)

ab_test_conversion = Counter(
    f"{settings.MONITORING.METRICS_PREFIX}_ab_test_conversion_total",
    "Total number of A/B test conversions",
    ["experiment_id", "variant"]
)

ab_test_allocation = Gauge(
    f"{settings.MONITORING.METRICS_PREFIX}_ab_test_allocation",
    "Current traffic allocation for A/B test variants",
    ["experiment_id", "variant"]
)


class ABTest:
    """A/B test implementation for model experimentation."""
    
    def __init__(
            self,
            experiment_id: str,
            variants: Dict[str, Any],
            traffic_allocation: Optional[Dict[str, float]] = None,
            allocation_strategy: Union[str, AllocationStrategy] = AllocationStrategy.RANDOM,
            description: str = "",
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None,
            feature_selector: Optional[Callable[[Dict[str, Any]], str]] = None
        ):
        """Initialize an A/B test.
        
        Args:
            experiment_id: Unique identifier for the experiment
            variants: Dictionary of variant names to variant implementations
            traffic_allocation: Dictionary of variant names to traffic percentages (0-1)
            allocation_strategy: Strategy for allocating traffic
            description: Description of the experiment
            start_time: When to start the experiment
            end_time: When to end the experiment
            feature_selector: Function to select variant based on features
        """
        self.experiment_id = experiment_id
        self.variants = variants
        self.description = description
        
        # Validate traffic allocation
        if traffic_allocation is None:
            # Equal distribution by default
            variant_count = len(variants)
            self.traffic_allocation = {v: 1.0 / variant_count for v in variants.keys()}
        else:
            # Validate that allocations sum to 1.0
            total_allocation = sum(traffic_allocation.values())
            if not np.isclose(total_allocation, 1.0):
                raise ValueError(f"Traffic allocation must sum to 1.0, got {total_allocation}")
            
            self.traffic_allocation = traffic_allocation
        
        # Set allocation strategy
        if isinstance(allocation_strategy, str):
            self.allocation_strategy = AllocationStrategy(allocation_strategy)
        else:
            self.allocation_strategy = allocation_strategy
        
        # Additional settings
        self.start_time = start_time or datetime.now()
        self.end_time = end_time
        self.status = ExperimentStatus.CREATED
        self.feature_selector = feature_selector
        
        # Tracking data
        self.metrics = {variant: {} for variant in variants.keys()}
        self.exposures = {variant: 0 for variant in variants.keys()}
        self.conversions = {variant: 0 for variant in variants.keys()}
        
        # Update Prometheus metrics
        for variant, allocation in self.traffic_allocation.items():
            ab_test_allocation.labels(
                experiment_id=self.experiment_id,
                variant=variant
            ).set(allocation)
        
        logger.info(f"Created A/B test experiment '{experiment_id}' with {len(variants)} variants")
    
    def start(self):
        """Start the A/B test experiment."""
        if self.status == ExperimentStatus.CREATED or self.status == ExperimentStatus.PAUSED:
            self.status = ExperimentStatus.RUNNING
            logger.info(f"Started A/B test experiment '{self.experiment_id}'")
        else:
            logger.warning(
                f"Cannot start experiment '{self.experiment_id}' with status {self.status.value}"
            )
    
    def pause(self):
        """Pause the A/B test experiment."""
        if self.status == ExperimentStatus.RUNNING:
            self.status = ExperimentStatus.PAUSED
            logger.info(f"Paused A/B test experiment '{self.experiment_id}'")
        else:
            logger.warning(
                f"Cannot pause experiment '{self.experiment_id}' with status {self.status.value}"
            )
    
    def resume(self):
        """Resume the A/B test experiment."""
        if self.status == ExperimentStatus.PAUSED:
            self.status = ExperimentStatus.RUNNING
            logger.info(f"Resumed A/B test experiment '{self.experiment_id}'")
        else:
            logger.warning(
                f"Cannot resume experiment '{self.experiment_id}' with status {self.status.value}"
            )
    
    def stop(self):
        """Stop the A/B test experiment."""
        if self.status != ExperimentStatus.STOPPED and self.status != ExperimentStatus.COMPLETED:
            self.status = ExperimentStatus.STOPPED
            logger.info(f"Stopped A/B test experiment '{self.experiment_id}'")
        else:
            logger.warning(
                f"Cannot stop experiment '{self.experiment_id}' with status {self.status.value}"
            )
    
    def complete(self):
        """Mark the A/B test experiment as completed."""
        self.status = ExperimentStatus.COMPLETED
        logger.info(f"Completed A/B test experiment '{self.experiment_id}'")
    
    def get_variant(
            self, 
            user_id: Optional[str] = None,
            features: Optional[Dict[str, Any]] = None
        ) -> str:
        """Get the variant to use for a request.
        
        Args:
            user_id: Identifier for the user
            features: Feature values for the request
            
        Returns:
            Selected variant name
        """
        if self.status != ExperimentStatus.RUNNING:
            # Default to first variant if not running
            variant = next(iter(self.variants.keys()))
            logger.warning(
                f"Experiment '{self.experiment_id}' is not running (status: {self.status.value}), "
                f"defaulting to variant '{variant}'"
            )
            return variant
        
        # Check if experiment has ended
        if self.end_time and datetime.now() > self.end_time:
            self.complete()
            # Default to first variant if completed
            variant = next(iter(self.variants.keys()))
            logger.info(
                f"Experiment '{self.experiment_id}' has ended, defaulting to variant '{variant}'"
            )
            return variant
        
        # Allocate variant based on strategy
        start_time = time.time()
        try:
            if self.allocation_strategy == AllocationStrategy.STICKY and user_id:
                variant = self._sticky_allocation(user_id)
            elif self.allocation_strategy == AllocationStrategy.TIME_BASED:
                variant = self._time_based_allocation()
            elif self.allocation_strategy == AllocationStrategy.FEATURE_BASED and features:
                variant = self._feature_based_allocation(features)
            else:
                variant = self._random_allocation()
            
            # Record exposure
            self.record_exposure(variant)
            
            # Record metrics
            ab_test_requests.labels(
                experiment_id=self.experiment_id,
                variant=variant
            ).inc()
            
            ab_test_latency.labels(
                experiment_id=self.experiment_id,
                variant=variant
            ).observe(time.time() - start_time)
            
            return variant
        except Exception as e:
            # Log error and default to first variant
            logger.error(f"Error getting variant for experiment '{self.experiment_id}': {str(e)}")
            variant = next(iter(self.variants.keys()))
            
            ab_test_errors.labels(
                experiment_id=self.experiment_id,
                variant=variant,
                error_type="allocation_error"
            ).inc()
            
            return variant
    
    def _random_allocation(self) -> str:
        """Allocate variant based on random selection.
        
        Returns:
            Selected variant name
        """
        # Get random number between 0 and 1
        r = random.random()
        
        # Find the variant that corresponds to this random number
        cumulative = 0.0
        for variant, allocation in self.traffic_allocation.items():
            cumulative += allocation
            if r <= cumulative:
                return variant
        
        # Fallback to last variant
        return list(self.variants.keys())[-1]
    
    def _sticky_allocation(self, user_id: str) -> str:
        """Allocate variant based on user ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            Selected variant name
        """
        # Hash the user ID to get a deterministic value
        hash_value = hash(user_id) % 1000 / 1000.0
        
        # Use the hash value for allocation
        cumulative = 0.0
        for variant, allocation in self.traffic_allocation.items():
            cumulative += allocation
            if hash_value <= cumulative:
                return variant
        
        # Fallback to last variant
        return list(self.variants.keys())[-1]
    
    def _time_based_allocation(self) -> str:
        """Allocate variant based on time of day.
        
        Returns:
            Selected variant name
        """
        # Get current hour (0-23)
        current_hour = datetime.now().hour
        
        # Simple time-based allocation: different variants for different times of day
        variants = list(self.variants.keys())
        variant_count = len(variants)
        
        # Divide the day into equal segments
        segment_size = 24 / variant_count
        segment_index = int(current_hour / segment_size)
        
        # Ensure index is valid
        segment_index = min(segment_index, variant_count - 1)
        
        return variants[segment_index]
    
    def _feature_based_allocation(self, features: Dict[str, Any]) -> str:
        """Allocate variant based on request features.
        
        Args:
            features: Feature values for the request
            
        Returns:
            Selected variant name
        """
        if self.feature_selector:
            try:
                # Use custom selector function
                return self.feature_selector(features)
            except Exception as e:
                logger.error(f"Error in feature selector: {str(e)}")
                # Fall back to random allocation
                return self._random_allocation()
        
        # Fallback to random allocation
        return self._random_allocation()
    
    def record_exposure(self, variant: str):
        """Record an exposure to a variant.
        
        Args:
            variant: Name of the variant
        """
        if variant not in self.exposures:
            logger.warning(f"Unknown variant '{variant}' for experiment '{self.experiment_id}'")
            return
        
        self.exposures[variant] += 1
    
    def record_conversion(self, variant: str):
        """Record a conversion for a variant.
        
        Args:
            variant: Name of the variant
        """
        if variant not in self.conversions:
            logger.warning(f"Unknown variant '{variant}' for experiment '{self.experiment_id}'")
            return
        
        self.conversions[variant] += 1
        
        # Update Prometheus metrics
        ab_test_conversion.labels(
            experiment_id=self.experiment_id,
            variant=variant
        ).inc()
    
    def record_metric(self, variant: str, metric_name: str, value: float):
        """Record a custom metric for a variant.
        
        Args:
            variant: Name of the variant
            metric_name: Name of the metric
            value: Metric value
        """
        if variant not in self.metrics:
            logger.warning(f"Unknown variant '{variant}' for experiment '{self.experiment_id}'")
            return
        
        if metric_name not in self.metrics[variant]:
            self.metrics[variant][metric_name] = []
        
        self.metrics[variant][metric_name].append(value)
    
    def get_conversion_rates(self) -> Dict[str, float]:
        """Calculate conversion rates for each variant.
        
        Returns:
            Dictionary of variant names to conversion rates
        """
        conversion_rates = {}
        
        for variant in self.variants.keys():
            exposures = self.exposures.get(variant, 0)
            conversions = self.conversions.get(variant, 0)
            
            if exposures > 0:
                rate = conversions / exposures
            else:
                rate = 0.0
            
            conversion_rates[variant] = rate
        
        return conversion_rates
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get a summary of metrics for each variant.
        
        Returns:
            Dictionary of metrics summary
        """
        summary = {}
        
        for variant in self.variants.keys():
            variant_metrics = {}
            
            for metric_name, values in self.metrics.get(variant, {}).items():
                if values:
                    variant_metrics[metric_name] = {
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "std": float(np.std(values)) if len(values) > 1 else 0.0,
                        "count": len(values)
                    }
            
            summary[variant] = variant_metrics
        
        return summary
    
    def get_results(self) -> Dict[str, Any]:
        """Get the results of the A/B test.
        
        Returns:
            Dictionary with test results
        """
        conversion_rates = self.get_conversion_rates()
        metrics_summary = self.get_metrics_summary()
        
        # Find the variant with the highest conversion rate
        if conversion_rates:
            best_variant = max(conversion_rates.items(), key=lambda x: x[1])
        else:
            best_variant = (next(iter(self.variants.keys())), 0.0)
        
        results = {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "variants": list(self.variants.keys()),
            "traffic_allocation": self.traffic_allocation,
            "exposures": self.exposures,
            "conversions": self.conversions,
            "conversion_rates": conversion_rates,
            "metrics": metrics_summary,
            "best_variant": {
                "name": best_variant[0],
                "conversion_rate": best_variant[1]
            },
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_time": datetime.now().isoformat()
        }
        
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the A/B test to a dictionary.
        
        Returns:
            Dictionary representation of the A/B test
        """
        return {
            "experiment_id": self.experiment_id,
            "description": self.description,
            "variants": list(self.variants.keys()),
            "traffic_allocation": self.traffic_allocation,
            "allocation_strategy": self.allocation_strategy.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "exposures": sum(self.exposures.values()),
            "conversions": sum(self.conversions.values())
        }


class ABTestManager:
    """Manager for A/B tests."""
    
    def __init__(self):
        """Initialize the A/B test manager."""
        self.experiments = {}
        self.default_experiment_id = None
    
    def create_experiment(
            self,
            experiment_id: str,
            variants: Dict[str, Any],
            traffic_allocation: Optional[Dict[str, float]] = None,
            allocation_strategy: Union[str, AllocationStrategy] = AllocationStrategy.RANDOM,
            description: str = "",
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None,
            feature_selector: Optional[Callable[[Dict[str, Any]], str]] = None,
            set_as_default: bool = False,
            auto_start: bool = False
        ) -> ABTest:
        """Create a new A/B test experiment.
        
        Args:
            experiment_id: Unique identifier for the experiment
            variants: Dictionary of variant names to variant implementations
            traffic_allocation: Dictionary of variant names to traffic percentages (0-1)
            allocation_strategy: Strategy for allocating traffic
            description: Description of the experiment
            start_time: When to start the experiment
            end_time: When to end the experiment
            feature_selector: Function to select variant based on features
            set_as_default: Whether to set this as the default experiment
            auto_start: Whether to automatically start the experiment
            
        Returns:
            Created A/B test experiment
        """
        if experiment_id in self.experiments:
            logger.warning(f"Experiment with ID '{experiment_id}' already exists, overwriting")
        
        experiment = ABTest(
            experiment_id=experiment_id,
            variants=variants,
            traffic_allocation=traffic_allocation,
            allocation_strategy=allocation_strategy,
            description=description,
            start_time=start_time,
            end_time=end_time,
            feature_selector=feature_selector
        )
        
        self.experiments[experiment_id] = experiment
        
        if set_as_default:
            self.default_experiment_id = experiment_id
        
        if auto_start:
            experiment.start()
        
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[ABTest]:
        """Get an experiment by ID.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            The experiment, or None if not found
        """
        return self.experiments.get(experiment_id)
    
    def get_default_experiment(self) -> Optional[ABTest]:
        """Get the default experiment.
        
        Returns:
            The default experiment, or None if not set
        """
        if self.default_experiment_id:
            return self.experiments.get(self.default_experiment_id)
        return None
    
    def set_default_experiment(self, experiment_id: str):
        """Set the default experiment.
        
        Args:
            experiment_id: ID of the experiment to set as default
        """
        if experiment_id in self.experiments:
            self.default_experiment_id = experiment_id
            logger.info(f"Set experiment '{experiment_id}' as default")
        else:
            logger.warning(f"Cannot set unknown experiment '{experiment_id}' as default")
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments.
        
        Returns:
            List of experiment dictionaries
        """
        return [exp.to_dict() for exp in self.experiments.values()]
    
    def start_experiment(self, experiment_id: str):
        """Start an experiment.
        
        Args:
            experiment_id: ID of the experiment to start
        """
        experiment = self.get_experiment(experiment_id)
        if experiment:
            experiment.start()
        else:
            logger.warning(f"Cannot start unknown experiment '{experiment_id}'")
    
    def stop_experiment(self, experiment_id: str):
        """Stop an experiment.
        
        Args:
            experiment_id: ID of the experiment to stop
        """
        experiment = self.get_experiment(experiment_id)
        if experiment:
            experiment.stop()
        else:
            logger.warning(f"Cannot stop unknown experiment '{experiment_id}'")
    
    def complete_experiment(self, experiment_id: str):
        """Mark an experiment as completed.
        
        Args:
            experiment_id: ID of the experiment to complete
        """
        experiment = self.get_experiment(experiment_id)
        if experiment:
            experiment.complete()
        else:
            logger.warning(f"Cannot complete unknown experiment '{experiment_id}'")
    
    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get the results of an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dictionary with experiment results
        """
        experiment = self.get_experiment(experiment_id)
        if experiment:
            return experiment.get_results()
        logger.warning(f"Cannot get results for unknown experiment '{experiment_id}'")
        return None
    
    def get_variant(
            self,
            experiment_id: Optional[str] = None,
            user_id: Optional[str] = None,
            features: Optional[Dict[str, Any]] = None
        ) -> Tuple[str, str]:
        """Get the variant to use for a request.
        
        Args:
            experiment_id: ID of the experiment (uses default if None)
            user_id: Identifier for the user
            features: Feature values for the request
            
        Returns:
            Tuple of (experiment_id, variant_name)
        """
        # If no experiment ID provided, use default
        if experiment_id is None:
            experiment = self.get_default_experiment()
            if experiment is None:
                logger.warning("No default experiment set, cannot get variant")
                return ("none", "none")
            experiment_id = experiment.experiment_id
        else:
            experiment = self.get_experiment(experiment_id)
            if experiment is None:
                logger.warning(f"Unknown experiment '{experiment_id}', cannot get variant")
                return ("none", "none")
        
        # Get variant
        variant = experiment.get_variant(user_id=user_id, features=features)
        return (experiment_id, variant)
    
    def record_conversion(
            self,
            experiment_id: str,
            variant: str
        ):
        """Record a conversion for a variant.
        
        Args:
            experiment_id: ID of the experiment
            variant: Name of the variant
        """
        experiment = self.get_experiment(experiment_id)
        if experiment:
            experiment.record_conversion(variant)
        else:
            logger.warning(f"Cannot record conversion for unknown experiment '{experiment_id}'")
    
    def record_metric(
            self,
            experiment_id: str,
            variant: str,
            metric_name: str,
            value: float
        ):
        """Record a custom metric for a variant.
        
        Args:
            experiment_id: ID of the experiment
            variant: Name of the variant
            metric_name: Name of the metric
            value: Metric value
        """
        experiment = self.get_experiment(experiment_id)
        if experiment:
            experiment.record_metric(variant, metric_name, value)
        else:
            logger.warning(f"Cannot record metric for unknown experiment '{experiment_id}'")


# Create a global instance of the AB test manager
ab_test_manager = ABTestManager()


def get_ab_test_manager() -> ABTestManager:
    """Get the global AB test manager.
    
    Returns:
        Global AB test manager instance
    """
    global ab_test_manager
    return ab_test_manager
