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

from mlops_production_system.config.settings import settings
from mlops_production_system.utils.logging_utils import get_logger

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
    
    def start(self) -> None:
        """Start the A/B test experiment."""
        self.status = ExperimentStatus.RUNNING
        logger.info(f"Started A/B test experiment '{self.experiment_id}'")
        
        # Start auto-stop timer if end_time is set
        if self.end_time:
            delay = (self.end_time - datetime.now()).total_seconds()
            if delay > 0:
                threading.Timer(delay, self.stop).start()
    
    def pause(self) -> None:
        """Pause the A/B test experiment."""
        self.status = ExperimentStatus.PAUSED
        logger.info(f"Paused A/B test experiment '{self.experiment_id}'")
    
    def resume(self) -> None:
        """Resume the A/B test experiment."""
        self.status = ExperimentStatus.RUNNING
        logger.info(f"Resumed A/B test experiment '{self.experiment_id}'")
    
    def stop(self) -> None:
        """Stop the A/B test experiment."""
        self.status = ExperimentStatus.STOPPED
        logger.info(f"Stopped A/B test experiment '{self.experiment_id}'")
    
    def complete(self) -> None:
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
            # Use first variant as default if experiment is not running
            return next(iter(self.variants.keys()))
        
        # Select variant based on allocation strategy
        if self.allocation_strategy == AllocationStrategy.RANDOM:
            # Random allocation based on traffic percentages
            return self._random_allocation()
        
        elif self.allocation_strategy == AllocationStrategy.STICKY:
            # Sticky allocation based on user ID
            if user_id is None:
                return self._random_allocation()
            return self._sticky_allocation(user_id)
        
        elif self.allocation_strategy == AllocationStrategy.TIME_BASED:
            # Time-based allocation
            return self._time_based_allocation()
        
        elif self.allocation_strategy == AllocationStrategy.FEATURE_BASED:
            # Feature-based allocation
            if features is None or self.feature_selector is None:
                return self._random_allocation()
            return self._feature_based_allocation(features)
        
        # Fallback to random allocation
        return self._random_allocation()
    
    def _random_allocation(self) -> str:
        """Allocate variant based on random selection.
        
        Returns:
            Selected variant name
        """
        variants = list(self.variants.keys())
        weights = [self.traffic_allocation[v] for v in variants]
        return random.choices(variants, weights=weights, k=1)[0]
    
    def _sticky_allocation(self, user_id: str) -> str:
        """Allocate variant based on user ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            Selected variant name
        """
        # Use hash of user_id to ensure consistent assignment
        variants = list(self.variants.keys())
        hash_value = hash(user_id) % 1000 / 1000.0  # Normalize to 0-1
        
        # Find the variant based on hash value and allocations
        cumulative = 0.0
        for variant in variants:
            cumulative += self.traffic_allocation[variant]
            if hash_value < cumulative:
                return variant
        
        # Fallback to last variant
        return variants[-1]
    
    def _time_based_allocation(self) -> str:
        """Allocate variant based on time of day.
        
        Returns:
            Selected variant name
        """
        # Simple time-based allocation using hour of day
        variants = list(self.variants.keys())
        hour = datetime.now().hour
        
        # Map hour to variant index
        variant_index = hour % len(variants)
        return variants[variant_index]
    
    def _feature_based_allocation(self, features: Dict[str, Any]) -> str:
        """Allocate variant based on request features.
        
        Args:
            features: Feature values for the request
            
        Returns:
            Selected variant name
        """
        # Use the provided feature selector function
        return self.feature_selector(features)
    
    def record_exposure(self, variant: str) -> None:
        """Record an exposure to a variant.
        
        Args:
            variant: Name of the variant
        """
        if variant not in self.variants:
            logger.warning(f"Unknown variant '{variant}' in experiment '{self.experiment_id}'")
            return
        
        self.exposures[variant] += 1
        ab_test_requests.labels(
            experiment_id=self.experiment_id,
            variant=variant
        ).inc()
    
    def record_conversion(self, variant: str) -> None:
        """Record a conversion for a variant.
        
        Args:
            variant: Name of the variant
        """
        if variant not in self.variants:
            logger.warning(f"Unknown variant '{variant}' in experiment '{self.experiment_id}'")
            return
        
        self.conversions[variant] += 1
        ab_test_conversion.labels(
            experiment_id=self.experiment_id,
            variant=variant
        ).inc()
    
    def record_metric(self, variant: str, metric_name: str, value: float) -> None:
        """Record a custom metric for a variant.
        
        Args:
            variant: Name of the variant
            metric_name: Name of the metric
            value: Metric value
        """
        if variant not in self.variants:
            logger.warning(f"Unknown variant '{variant}' in experiment '{self.experiment_id}'")
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
            if self.exposures[variant] == 0:
                conversion_rates[variant] = 0.0
            else:
                conversion_rates[variant] = self.conversions[variant] / self.exposures[variant]
        
        return conversion_rates
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get a summary of metrics for each variant.
        
        Returns:
            Dictionary of metrics summary
        """
        summary = {}
        for variant in self.variants.keys():
            summary[variant] = {}
            for metric_name, values in self.metrics[variant].items():
                if values:
                    summary[variant][metric_name] = {
                        "mean": np.mean(values),
                        "median": np.median(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "std": np.std(values),
                        "count": len(values)
                    }
        
        return summary
    
    def get_results(self) -> Dict[str, Any]:
        """Get the results of the A/B test.
        
        Returns:
            Dictionary with test results
        """
        conversion_rates = self.get_conversion_rates()
        metrics_summary = self.get_metrics_summary()
        
        # Calculate confidence intervals and statistical significance
        # (simplified implementation)
        statistical_significance = {}
        for variant in self.variants.keys():
            # Compare each variant to the first variant (control)
            if variant == next(iter(self.variants.keys())):
                statistical_significance[variant] = True  # Control is significant with itself
                continue
            
            # Simple significance check based on exposures
            # In a real implementation, this would use proper statistical tests
            statistical_significance[variant] = self.exposures[variant] >= 100
        
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "variants": list(self.variants.keys()),
            "traffic_allocation": self.traffic_allocation,
            "exposures": self.exposures,
            "conversions": self.conversions,
            "conversion_rates": conversion_rates,
            "metrics": metrics_summary,
            "statistical_significance": statistical_significance,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds()
        }
    
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
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None
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
        if self.default_experiment_id is None:
            return None
        return self.get_experiment(self.default_experiment_id)
    
    def set_default_experiment(self, experiment_id: str) -> None:
        """Set the default experiment.
        
        Args:
            experiment_id: ID of the experiment to set as default
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment '{experiment_id}' does not exist")
        
        self.default_experiment_id = experiment_id
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments.
        
        Returns:
            List of experiment dictionaries
        """
        return [experiment.to_dict() for experiment in self.experiments.values()]
    
    def start_experiment(self, experiment_id: str) -> None:
        """Start an experiment.
        
        Args:
            experiment_id: ID of the experiment to start
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_id}' does not exist")
        
        experiment.start()
    
    def stop_experiment(self, experiment_id: str) -> None:
        """Stop an experiment.
        
        Args:
            experiment_id: ID of the experiment to stop
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_id}' does not exist")
        
        experiment.stop()
    
    def complete_experiment(self, experiment_id: str) -> None:
        """Mark an experiment as completed.
        
        Args:
            experiment_id: ID of the experiment to complete
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_id}' does not exist")
        
        experiment.complete()
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get the results of an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dictionary with experiment results
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_id}' does not exist")
        
        return experiment.get_results()
    
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
        # Use default experiment if not specified
        if experiment_id is None:
            experiment = self.get_default_experiment()
            if experiment is None:
                raise ValueError("No default experiment set")
            experiment_id = experiment.experiment_id
        else:
            experiment = self.get_experiment(experiment_id)
            if experiment is None:
                raise ValueError(f"Experiment '{experiment_id}' does not exist")
        
        # Get variant and record exposure
        variant = experiment.get_variant(user_id=user_id, features=features)
        experiment.record_exposure(variant)
        
        return experiment_id, variant
    
    def record_conversion(
        self,
        experiment_id: str,
        variant: str
    ) -> None:
        """Record a conversion for a variant.
        
        Args:
            experiment_id: ID of the experiment
            variant: Name of the variant
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_id}' does not exist")
        
        experiment.record_conversion(variant)
    
    def record_metric(
        self,
        experiment_id: str,
        variant: str,
        metric_name: str,
        value: float
    ) -> None:
        """Record a custom metric for a variant.
        
        Args:
            experiment_id: ID of the experiment
            variant: Name of the variant
            metric_name: Name of the metric
            value: Metric value
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_id}' does not exist")
        
        experiment.record_metric(variant, metric_name, value)


# Create a global instance of the AB test manager
ab_test_manager = ABTestManager()


def get_ab_test_manager() -> ABTestManager:
    """Get the global AB test manager.
    
    Returns:
        Global AB test manager instance
    """
    return ab_test_manager
