"""MLflow extensions and custom plugins for MLOps Production System."""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from functools import wraps
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run, Experiment
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from mlops_production_system.config.settings import settings
from mlops_production_system.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ModelRegistryIntegration:
    """Integration with MLflow Model Registry."""

    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize the integration.
        
        Args:
            tracking_uri: MLflow tracking URI
        """
        self.tracking_uri = tracking_uri or settings.MLFLOW.TRACKING_URI
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)
        
    def register_model(
        self, 
        run_id: str, 
        model_name: str, 
        artifact_path: str = "model"
    ) -> str:
        """Register a model in the MLflow Model Registry.
        
        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            artifact_path: Path to the model artifact
            
        Returns:
            Model version
        """
        model_details = mlflow.register_model(
            model_uri=f"runs:/{run_id}/{artifact_path}",
            name=model_name
        )
        return model_details.version
    
    def promote_model(
        self, 
        model_name: str, 
        version: str, 
        stage: str
    ) -> None:
        """Promote a model to a new stage.
        
        Args:
            model_name: Name of the registered model
            version: Model version
            stage: Target stage (Staging, Production, Archived)
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
    def get_latest_model(
        self, 
        model_name: str, 
        stage: str = "Production"
    ) -> Dict[str, Any]:
        """Get the latest model from the registry.
        
        Args:
            model_name: Name of the registered model
            stage: Model stage to filter by
            
        Returns:
            Dictionary with model information
        """
        models = self.client.get_latest_versions(model_name, stages=[stage])
        if not models:
            raise ValueError(f"No models found for {model_name} in {stage} stage")
        
        model = models[0]
        return {
            "name": model.name,
            "version": model.version,
            "stage": model.current_stage,
            "run_id": model.run_id,
            "creation_timestamp": model.creation_timestamp,
            "uri": f"models:/{model.name}/{model.version}"
        }


class ExperimentTracker:
    """Enhanced experiment tracking with MLflow."""

    def __init__(
        self, 
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """Initialize the experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI
            tags: Tags to set on the experiment
        """
        self.tracking_uri = tracking_uri or settings.MLFLOW.TRACKING_URI
        self.experiment_name = experiment_name or settings.MLFLOW.EXPERIMENT_NAME
        self.tags = tags or {}
        
        # Set up MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                name=self.experiment_name,
                tags=self.tags
            )
        else:
            self.experiment_id = experiment.experiment_id
            
        # Initialize client
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
    def start_run(
        self, 
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Tags to set on the run
            
        Returns:
            Run ID
        """
        tags = tags or {}
        
        # Add default tags
        tags.update({
            "user": os.environ.get("USER", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "mlops_version": settings.VERSION
        })
        
        # Start run
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags
        )
        
        return run.info.run_id
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
    
    def log_model_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        prefix: str = ""
    ) -> Dict[str, float]:
        """Log comprehensive model performance metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (for classification)
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, mean_squared_error,
            mean_absolute_error, r2_score
        )
        
        metrics = {}
        
        # Determine if classification or regression
        if len(np.unique(y_true)) < 10:  # Heuristic for classification
            # Classification metrics
            metrics[f"{prefix}accuracy"] = accuracy_score(y_true, y_pred)
            
            try:
                # Handle binary vs multiclass
                if len(np.unique(y_true)) == 2:
                    metrics[f"{prefix}precision"] = precision_score(y_true, y_pred)
                    metrics[f"{prefix}recall"] = recall_score(y_true, y_pred)
                    metrics[f"{prefix}f1"] = f1_score(y_true, y_pred)
                    
                    if y_prob is not None:
                        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                            prob_scores = y_prob[:, 1]  # Probability of positive class
                        else:
                            prob_scores = y_prob
                            
                        metrics[f"{prefix}roc_auc"] = roc_auc_score(y_true, prob_scores)
                        
                        # Log ROC curve as a figure
                        self.log_roc_curve(y_true, prob_scores, prefix)
                        
                        # Log precision-recall curve
                        self.log_pr_curve(y_true, prob_scores, prefix)
                else:
                    # Multiclass
                    metrics[f"{prefix}precision_macro"] = precision_score(y_true, y_pred, average='macro')
                    metrics[f"{prefix}recall_macro"] = recall_score(y_true, y_pred, average='macro')
                    metrics[f"{prefix}f1_macro"] = f1_score(y_true, y_pred, average='macro')
                    
                    if y_prob is not None:
                        metrics[f"{prefix}roc_auc_ovr"] = roc_auc_score(
                            y_true, y_prob, multi_class='ovr', average='macro'
                        )
            except Exception as e:
                logger.warning(f"Error calculating some classification metrics: {str(e)}")
                
            # Log confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            mlflow.log_text(
                json.dumps(cm.tolist()),
                f"{prefix}confusion_matrix.json"
            )
            
        else:
            # Regression metrics
            metrics[f"{prefix}mse"] = mean_squared_error(y_true, y_pred)
            metrics[f"{prefix}rmse"] = np.sqrt(metrics[f"{prefix}mse"])
            metrics[f"{prefix}mae"] = mean_absolute_error(y_true, y_pred)
            metrics[f"{prefix}r2"] = r2_score(y_true, y_pred)
            
            # Log residuals plot
            self.log_residuals_plot(y_true, y_pred, prefix)
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        return metrics
    
    def log_roc_curve(
        self, 
        y_true: np.ndarray, 
        y_score: np.ndarray,
        prefix: str = ""
    ) -> None:
        """Log ROC curve as a figure.
        
        Args:
            y_true: Ground truth labels
            y_score: Prediction scores
            prefix: Prefix for artifact name
        """
        import matplotlib.pyplot as plt
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Save to temp file and log
        temp_path = f"/tmp/{prefix}roc_curve.png"
        plt.savefig(temp_path)
        mlflow.log_artifact(temp_path, "curves")
        os.remove(temp_path)
        plt.close()
    
    def log_pr_curve(
        self, 
        y_true: np.ndarray, 
        y_score: np.ndarray,
        prefix: str = ""
    ) -> None:
        """Log precision-recall curve as a figure.
        
        Args:
            y_true: Ground truth labels
            y_score: Prediction scores
            prefix: Prefix for artifact name
        """
        import matplotlib.pyplot as plt
        
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve')
        
        # Save to temp file and log
        temp_path = f"/tmp/{prefix}pr_curve.png"
        plt.savefig(temp_path)
        mlflow.log_artifact(temp_path, "curves")
        os.remove(temp_path)
        plt.close()
    
    def log_residuals_plot(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> None:
        """Log residuals plot for regression.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            prefix: Prefix for artifact name
        """
        import matplotlib.pyplot as plt
        
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        # Save to temp file and log
        temp_path = f"/tmp/{prefix}residuals_plot.png"
        plt.savefig(temp_path)
        mlflow.log_artifact(temp_path, "plots")
        os.remove(temp_path)
        plt.close()
    
    def log_feature_importance(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        importance_type: str = "feature_importance"
    ) -> None:
        """Log feature importance.
        
        Args:
            feature_names: List of feature names
            importance_values: Feature importance values
            importance_type: Type of importance (e.g., "feature_importance", "shap")
        """
        # Create dataframe for visualization
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance_values
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values("importance", ascending=False)
        
        # Log as JSON
        mlflow.log_text(
            importance_df.to_json(orient="records"),
            f"{importance_type}.json"
        )
        
        # Create and log plot
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        plt.barh(
            importance_df["feature"].values[:20],  # Top 20 features
            importance_df["importance"].values[:20],
            color="skyblue"
        )
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Feature Importance ({importance_type})")
        plt.tight_layout()
        
        # Save to temp file and log
        temp_path = f"/tmp/{importance_type}_plot.png"
        plt.savefig(temp_path)
        mlflow.log_artifact(temp_path, "feature_importance")
        os.remove(temp_path)
        plt.close()


def track_experiment(experiment_name: Optional[str] = None):
    """Decorator to track an experiment with MLflow.
    
    Args:
        experiment_name: Name of the MLflow experiment
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set up experiment tracking
            tracker = ExperimentTracker(experiment_name=experiment_name)
            
            # Extract function name and arguments for run name
            func_name = func.__name__
            run_name = f"{func_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Start run and log parameters
            tracker.start_run(run_name=run_name)
            
            # Log non-object function arguments
            params = {}
            for k, v in kwargs.items():
                if isinstance(v, (str, int, float, bool)):
                    params[k] = v
                elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in v):
                    params[k] = str(v)
            
            mlflow.log_params(params)
            
            try:
                # Run the function
                result = func(*args, **kwargs)
                
                # Log result if it's a dict with metrics
                if isinstance(result, dict):
                    metrics = {k: v for k, v in result.items() 
                              if isinstance(v, (int, float)) and not isinstance(v, bool)}
                    if metrics:
                        mlflow.log_metrics(metrics)
                
                return result
            except Exception as e:
                # Log the error
                mlflow.set_tag("error", str(e))
                mlflow.log_text(str(e), "error.txt")
                raise
            finally:
                # End the run
                tracker.end_run()
                
        return wrapper
    return decorator
