#!/usr/bin/env python
"""Model validation script for CI/CD pipeline.

This script validates ML models before deployment to ensure they meet quality standards.
It's designed to be used in a CI/CD pipeline as a quality gate before promoting models
to production.

Usage:
    python model_validation.py --model-uri <model_uri> --data-path <validation_data_path>
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, Optional, List, Tuple

import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_validation_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load validation data from file.
    
    Args:
        data_path: Path to validation data CSV
        
    Returns:
        Tuple of features and target
    """
    # Load the data
    logger.info(f"Loading validation data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Validate the data has the target column
    if "target" not in df.columns:
        raise ValueError("Validation data must contain a 'target' column")
    
    # Split features and target
    X = df.drop("target", axis=1)
    y = df["target"]
    
    logger.info(f"Loaded validation data with {X.shape[0]} samples and {X.shape[1]} features")
    return X, y


def load_model(model_uri: str):
    """Load model from MLflow model registry or local path.
    
    Args:
        model_uri: URI of the model (MLflow model URI format)
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold_config: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Validate model performance against thresholds.
    
    Args:
        model: Loaded model
        X: Validation features
        y: Validation target
        threshold_config: Configuration with metric thresholds
        
    Returns:
        Dictionary with validation results
    """
    # Set default thresholds if not provided
    if threshold_config is None:
        threshold_config = {
            "accuracy": 0.7,
            "f1": 0.7,
            "precision": 0.7,
            "recall": 0.7,
            "mse": 100,  # For regression
            "mae": 10,   # For regression
            "r2": 0.5    # For regression
        }
    
    # Make predictions
    logger.info("Making predictions on validation data")
    predictions = model.predict(X)
    
    # Determine task type (classification or regression)
    num_unique_values = len(np.unique(y))
    is_classification = num_unique_values < 10  # Heuristic
    
    # Calculate metrics
    metrics = {}
    validation_results = {
        "passed": True,
        "metrics": {},
        "threshold_metrics": {},
        "failures": []
    }
    
    if is_classification:
        # For classification, calculate classification metrics
        logger.info("Calculating classification metrics")
        
        # For predicted probabilities (if available)
        try:
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X)
                if probas.shape[1] == 2:  # Binary classification
                    roc_auc = roc_auc_score(y, probas[:, 1])
                    metrics["roc_auc"] = roc_auc
                    validation_results["metrics"]["roc_auc"] = roc_auc
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {str(e)}")
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y, predictions)
        
        # For binary classification
        if num_unique_values == 2:
            metrics["precision"] = precision_score(y, predictions, average="binary")
            metrics["recall"] = recall_score(y, predictions, average="binary")
            metrics["f1"] = f1_score(y, predictions, average="binary")
        else:
            # Multiclass
            metrics["precision"] = precision_score(y, predictions, average="weighted")
            metrics["recall"] = recall_score(y, predictions, average="weighted")
            metrics["f1"] = f1_score(y, predictions, average="weighted")
        
        # Check against thresholds
        for metric_name, metric_value in metrics.items():
            validation_results["metrics"][metric_name] = metric_value
            
            if metric_name in threshold_config:
                threshold = threshold_config[metric_name]
                validation_results["threshold_metrics"][metric_name] = threshold
                
                # Classification metrics should be above threshold
                if metric_value < threshold:
                    validation_results["passed"] = False
                    validation_results["failures"].append({
                        "metric": metric_name,
                        "value": metric_value,
                        "threshold": threshold,
                        "message": f"{metric_name} ({metric_value:.4f}) is below threshold ({threshold:.4f})"
                    })
    else:
        # For regression, calculate regression metrics
        logger.info("Calculating regression metrics")
        metrics["mse"] = mean_squared_error(y, predictions)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(y, predictions)
        metrics["r2"] = r2_score(y, predictions)
        
        # Check against thresholds
        for metric_name, metric_value in metrics.items():
            validation_results["metrics"][metric_name] = metric_value
            
            if metric_name in threshold_config:
                threshold = threshold_config[metric_name]
                validation_results["threshold_metrics"][metric_name] = threshold
                
                # For error metrics (lower is better)
                if metric_name in ["mse", "rmse", "mae"]:
                    if metric_value > threshold:
                        validation_results["passed"] = False
                        validation_results["failures"].append({
                            "metric": metric_name,
                            "value": metric_value,
                            "threshold": threshold,
                            "message": f"{metric_name} ({metric_value:.4f}) is above threshold ({threshold:.4f})"
                        })
                # For R² (higher is better)
                elif metric_name == "r2":
                    if metric_value < threshold:
                        validation_results["passed"] = False
                        validation_results["failures"].append({
                            "metric": metric_name,
                            "value": metric_value,
                            "threshold": threshold,
                            "message": f"{metric_name} ({metric_value:.4f}) is below threshold ({threshold:.4f})"
                        })
    
    # Log summary
    if validation_results["passed"]:
        logger.info("Model validation PASSED. All metrics meet thresholds.")
    else:
        logger.warning("Model validation FAILED. Some metrics did not meet thresholds.")
        for failure in validation_results["failures"]:
            logger.warning(f"  - {failure['message']}")
    
    return validation_results


def save_validation_results(results: Dict[str, Any], output_path: str) -> None:
    """Save validation results to a JSON file.
    
    Args:
        results: Validation results dictionary
        output_path: Path to save the results
    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Validation results saved to {output_path}")


def main():
    """Main function to run model validation."""
    parser = argparse.ArgumentParser(description="Validate ML model for deployment")
    parser.add_argument("--model-uri", required=True, help="URI of the model to validate")
    parser.add_argument("--data-path", required=True, help="Path to validation data")
    parser.add_argument("--threshold-config", help="Path to threshold configuration JSON")
    parser.add_argument("--output-path", default="validation_results.json", 
                       help="Path to save validation results")
    
    args = parser.parse_args()
    
    # Load threshold configuration if provided
    threshold_config = None
    if args.threshold_config:
        with open(args.threshold_config, "r") as f:
            threshold_config = json.load(f)
    
    try:
        # Load validation data
        X, y = load_validation_data(args.data_path)
        
        # Load model
        model = load_model(args.model_uri)
        
        # Validate model
        validation_results = validate_model(model, X, y, threshold_config)
        
        # Save validation results
        save_validation_results(validation_results, args.output_path)
        
        # Return success/failure
        if validation_results["passed"]:
            sys.exit(0)
        else:
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error during model validation: {str(e)}")
        sys.exit(2)


if __name__ == "__main__":
    main()
