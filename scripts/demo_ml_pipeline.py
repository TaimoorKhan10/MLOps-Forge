"""Demonstration script for the complete ML pipeline."""

import os
import argparse
import logging
from pathlib import Path

import pandas as pd
import mlflow

from mlops_production_system.data.data_loader import get_data_loader
from mlops_production_system.data.feature_engineering import get_feature_engineer
from mlops_production_system.models.model_trainer import get_model_trainer
from mlops_production_system.utils.logging_utils import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


def run_ml_pipeline(data_path, target_column, model_type, output_dir, test_size=0.2, val_size=0.1):
    """Run the complete ML pipeline from data loading to model evaluation.
    
    Args:
        data_path: Path to the data file.
        target_column: Name of the target column.
        model_type: Type of model to train.
        output_dir: Directory to save the model and artifacts.
        test_size: Fraction of data to use for testing.
        val_size: Fraction of training data to use for validation.
    """
    logger.info("Starting ML pipeline")
    
    # Step 1: Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 2: Load data
    logger.info(f"Loading data from {data_path}")
    data_loader = get_data_loader(data_path=os.path.dirname(data_path))
    data = data_loader.load_data(os.path.basename(data_path))
    
    logger.info(f"Loaded data with shape: {data.shape}")
    logger.info(f"Columns: {', '.join(data.columns)}")
    
    # Step 3: Split data
    logger.info("Splitting data into train, validation, and test sets")
    splits = data_loader.split_data(
        data=data,
        target_column=target_column,
        test_size=test_size,
        val_size=val_size,
        random_state=42
    )
    
    logger.info(f"Train set: {splits['X_train'].shape}")
    logger.info(f"Validation set: {splits['X_val'].shape}")
    logger.info(f"Test set: {splits['X_test'].shape}")
    
    # Step 4: Feature engineering
    logger.info("Performing feature engineering")
    feature_engineer = get_feature_engineer()
    
    # Identify categorical and numerical features
    categorical_features = splits['X_train'].select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = splits['X_train'].select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    logger.info(f"Categorical features: {categorical_features}")
    logger.info(f"Numerical features: {numerical_features}")
    
    # Fit and transform the data
    X_train_processed = feature_engineer.fit_transform(
        splits["X_train"],
        categorical_features=categorical_features,
        numerical_features=numerical_features
    )
    
    X_val_processed = feature_engineer.transform(splits["X_val"])
    X_test_processed = feature_engineer.transform(splits["X_test"])
    
    # Save feature metadata
    feature_metadata_path = os.path.join(output_dir, "feature_metadata.json")
    feature_engineer.save_features(feature_metadata_path)
    logger.info(f"Feature metadata saved to {feature_metadata_path}")
    
    # Step 5: Model training
    logger.info(f"Training {model_type} model")
    model_trainer = get_model_trainer(
        model_type=model_type,
        hyperparameters={"n_estimators": 100, "random_state": 42}
    )
    
    # Train with validation data
    model = model_trainer.train(
        X_train=X_train_processed,
        y_train=splits["y_train"],
        X_val=X_val_processed,
        y_val=splits["y_val"]
    )
    
    # Step 6: Model evaluation
    logger.info("Evaluating model on test data")
    test_metrics = model_trainer.evaluate(X_test_processed, splits["y_test"], prefix="test_")
    
    for metric_name, metric_value in test_metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # Step 7: Save model
    logger.info("Saving model")
    model_path = os.path.join(output_dir, f"{model_type}_model.joblib")
    model_trainer.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Step 8: Save model metadata
    model_metadata = {
        "model_name": model_type,
        "model_version": "0.1.0",
        "model_type": model_type,
        "created_at": pd.Timestamp.now().isoformat(),
        "features": feature_engineer.feature_names if hasattr(feature_engineer, "feature_names") else [],
        "metrics": test_metrics,
    }
    
    import json
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(model_metadata, f, indent=2)
    
    logger.info(f"Model metadata saved to {os.path.join(output_dir, 'metadata.json')}")
    logger.info("ML pipeline completed successfully")
    
    return model_path, test_metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run ML pipeline')
    parser.add_argument('--data', type=str, default='data/raw/sample_data.csv',
                        help='Path to the data file')
    parser.add_argument('--target', type=str, default='target',
                        help='Name of the target column')
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'logistic_regression'],
                        help='Type of model to train')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save the model and artifacts')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='Fraction of training data to use for validation')
    
    args = parser.parse_args()
    
    # Run ML pipeline
    model_path, test_metrics = run_ml_pipeline(
        data_path=args.data,
        target_column=args.target,
        model_type=args.model_type,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    print("\n=== ML Pipeline Completed ===")
    print(f"Model saved to: {model_path}")
    print("\nTest Metrics:")
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print("\nNext steps:")
    print("1. Start the API server: mlops-serve")
    print("2. Start the monitoring server: mlops-monitor start-prometheus")
    print("3. Make predictions using the API")


if __name__ == "__main__":
    main()
