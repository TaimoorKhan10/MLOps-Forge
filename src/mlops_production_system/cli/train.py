"""Command-line interface for model training."""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler

from mlops_production_system.config.settings import settings
from mlops_production_system.data.data_loader import get_data_loader
from mlops_production_system.data.feature_engineering import get_feature_engineer
from mlops_production_system.models.model_trainer import get_model_trainer
from mlops_production_system.utils.logging_utils import setup_logging, get_logger

# Set up CLI app
app = typer.Typer(help="MLOps Production System - Model Training CLI")
console = Console()
logger = get_logger(__name__)


@app.command()
def train(
    data_path: str = typer.Option(
        None, "--data-path", "-d", help="Path to the training data file"
    ),
    target_column: str = typer.Option(
        ..., "--target-column", "-t", help="Name of the target column"
    ),
    model_type: str = typer.Option(
        "random_forest", "--model-type", "-m", 
        help="Type of model to train (random_forest, gradient_boosting, logistic_regression)"
    ),
    output_dir: str = typer.Option(
        None, "--output-dir", "-o", help="Directory to save the model and artifacts"
    ),
    test_size: float = typer.Option(
        0.2, "--test-size", help="Fraction of data to use for testing"
    ),
    val_size: float = typer.Option(
        0.1, "--val-size", help="Fraction of training data to use for validation"
    ),
    random_state: int = typer.Option(
        42, "--random-state", help="Random state for reproducibility"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l", help="Logging level"
    ),
):
    """Train a machine learning model and save it to disk."""
    # Set up logging
    setup_logging(level=log_level)
    
    try:
        # Determine paths
        data_path = data_path or os.path.join(settings.DATA.RAW_DATA_DIR, "data.csv")
        output_dir = output_dir or settings.MODEL.MODEL_DIR
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Log configuration
        logger.info(f"Training configuration:")
        logger.info(f"  Data path: {data_path}")
        logger.info(f"  Target column: {target_column}")
        logger.info(f"  Model type: {model_type}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Test size: {test_size}")
        logger.info(f"  Validation size: {val_size}")
        logger.info(f"  Random state: {random_state}")
        
        # Load data
        console.print("[bold green]Loading data...[/bold green]")
        data_loader = get_data_loader()
        
        # Extract file name from path
        file_name = os.path.basename(data_path)
        data_dir = os.path.dirname(data_path)
        
        # Set data path for data loader
        data_loader.data_path = data_dir
        
        # Load data
        data = data_loader.load_data(file_name)
        console.print(f"Loaded data with shape: {data.shape}")
        
        # Split data
        console.print("[bold green]Splitting data...[/bold green]")
        splits = data_loader.split_data(
            data=data,
            target_column=target_column,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            stratify=True
        )
        
        # Feature engineering
        console.print("[bold green]Engineering features...[/bold green]")
        feature_engineer = get_feature_engineer()
        
        # Drop target column from features
        X_train = splits["X_train"]
        y_train = splits["y_train"]
        
        # Fit feature engineering pipeline
        X_train_processed = feature_engineer.fit_transform(X_train)
        
        # Transform validation and test data if available
        X_val_processed = None
        X_test_processed = None
        
        if "X_val" in splits:
            X_val_processed = feature_engineer.transform(splits["X_val"])
        
        if "X_test" in splits:
            X_test_processed = feature_engineer.transform(splits["X_test"])
        
        # Save feature metadata
        feature_metadata_path = os.path.join(output_dir, "feature_metadata.json")
        feature_engineer.save_features(feature_metadata_path)
        
        # Train model
        console.print("[bold green]Training model...[/bold green]")
        model_trainer = get_model_trainer(model_type=model_type)
        
        # Train with validation data if available
        if X_val_processed is not None and "y_val" in splits:
            model = model_trainer.train(
                X_train=X_train_processed,
                y_train=y_train,
                X_val=X_val_processed,
                y_val=splits["y_val"]
            )
        else:
            model = model_trainer.train(X_train=X_train_processed, y_train=y_train)
        
        # Evaluate on test data if available
        if X_test_processed is not None and "y_test" in splits:
            console.print("[bold green]Evaluating model on test data...[/bold green]")
            test_metrics = model_trainer.evaluate(X_test_processed, splits["y_test"], prefix="test_")
            
            for metric_name, metric_value in test_metrics.items():
                console.print(f"{metric_name}: {metric_value:.4f}")
        
        # Save model
        console.print("[bold green]Saving model...[/bold green]")
        model_path = os.path.join(output_dir, f"{model_type}_model.joblib")
        model_trainer.save_model(model_path)
        
        # Save model metadata
        model_metadata = {
            "model_name": model_type,
            "model_version": settings.MODEL.MODEL_VERSION,
            "model_type": model_type,
            "created_at": pd.Timestamp.now().isoformat(),
            "features": feature_engineer.feature_names if hasattr(feature_engineer, "feature_names") else [],
            "metrics": test_metrics if "test_metrics" in locals() else {},
        }
        
        import json
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(model_metadata, f, indent=2)
        
        console.print(f"[bold green]Model saved to {model_path}[/bold green]")
        console.print(f"[bold green]Training completed successfully![/bold green]")
        
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        console.print(f"[bold red]Error during training: {e}[/bold red]")
        raise typer.Exit(code=1)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
