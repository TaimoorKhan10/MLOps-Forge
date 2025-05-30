"""Training pipeline implementation."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score

from mlops_production_system.config.settings import settings
from mlops_production_system.data.data_loader import get_data_loader
from mlops_production_system.data.feature_engineering import get_feature_engineer
from mlops_production_system.models.model_trainer import get_model_trainer
from mlops_production_system.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline for ML models."""

    def __init__(
        self,
        data_path: Optional[str] = None,
        target_column: str = "target",
        model_type: str = "random_forest",
        output_dir: Optional[str] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        """Initialize the training pipeline.
        
        Args:
            data_path: Path to the data file.
            target_column: Name of the target column.
            model_type: Type of model to train.
            output_dir: Directory to save model artifacts.
            test_size: Fraction of data to use for testing.
            val_size: Fraction of training data to use for validation.
            random_state: Random state for reproducibility.
        """
        self.data_path = data_path or os.path.join(settings.DATA.RAW_DATA_DIR, "sample_data.csv")
        self.target_column = target_column
        self.model_type = model_type
        self.output_dir = output_dir or settings.MODEL.MODEL_DIR
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.data_loader = None
        self.feature_engineer = None
        self.model_trainer = None
        self.model_path = None
        self.metrics = None
        
        # Set up MLflow
        if settings.MLFLOW.TRACKING_URI:
            mlflow.set_tracking_uri(settings.MLFLOW.TRACKING_URI)

    def run(self) -> Dict[str, float]:
        """Run the end-to-end training pipeline.
        
        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info(f"Starting training pipeline for {self.model_type} model")
        
        # Step 1: Load data
        logger.info(f"Loading data from {self.data_path}")
        self.data_loader = get_data_loader(data_path=os.path.dirname(self.data_path))
        data = self.data_loader.load_data(os.path.basename(self.data_path))
        
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Step 2: Split data
        logger.info("Splitting data into train, validation, and test sets")
        splits = self.data_loader.split_data(
            data=data,
            target_column=self.target_column,
            test_size=self.test_size,
            val_size=self.val_size,
            random_state=self.random_state
        )
        
        # Step 3: Feature engineering
        logger.info("Performing feature engineering")
        self.feature_engineer = get_feature_engineer()
        
        # Identify categorical and numerical features
        categorical_features = splits['X_train'].select_dtypes(
            include=['object', 'category']).columns.tolist()
        numerical_features = splits['X_train'].select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        
        logger.info(f"Categorical features: {categorical_features}")
        logger.info(f"Numerical features: {numerical_features}")
        
        # Fit and transform the data
        X_train_processed = self.feature_engineer.fit_transform(
            splits["X_train"],
            categorical_features=categorical_features,
            numerical_features=numerical_features
        )
        
        X_val_processed = self.feature_engineer.transform(splits["X_val"])
        X_test_processed = self.feature_engineer.transform(splits["X_test"])
        
        # Save feature metadata
        feature_metadata_path = os.path.join(self.output_dir, "feature_metadata.json")
        self.feature_engineer.save_features(feature_metadata_path)
        logger.info(f"Feature metadata saved to {feature_metadata_path}")
        
        # Step 4: Model training
        logger.info(f"Training {self.model_type} model")
        self.model_trainer = get_model_trainer(
            model_type=self.model_type,
            hyperparameters={"random_state": self.random_state}
        )
        
        # Train with validation data
        self.model_trainer.train(
            X_train=X_train_processed,
            y_train=splits["y_train"],
            X_val=X_val_processed,
            y_val=splits["y_val"]
        )
        
        # Step 5: Model evaluation
        logger.info("Evaluating model on test data")
        self.metrics = self.model_trainer.evaluate(
            X_test_processed, 
            splits["y_test"], 
            prefix="test_"
        )
        
        for metric_name, metric_value in self.metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")
        
        # Step 6: Save model
        logger.info("Saving model")
        self.model_path = os.path.join(self.output_dir, f"{self.model_type}_model.joblib")
        self.model_trainer.save_model(self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        # Step 7: Save model metadata
        model_metadata = {
            "model_name": self.model_type,
            "model_version": settings.MODEL.VERSION,
            "model_type": self.model_type,
            "created_at": pd.Timestamp.now().isoformat(),
            "features": self.feature_engineer.feature_names 
                if hasattr(self.feature_engineer, "feature_names") else [],
            "metrics": self.metrics,
        }
        
        import json
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(model_metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to {os.path.join(self.output_dir, 'metadata.json')}")
        logger.info("Training pipeline completed successfully")
        
        return self.metrics


def run_training_pipeline(
    data_path: Optional[str] = None,
    target_column: str = "target",
    model_type: str = "random_forest",
    output_dir: Optional[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, float]:
    """Run the training pipeline.
    
    Args:
        data_path: Path to the data file.
        target_column: Name of the target column.
        model_type: Type of model to train.
        output_dir: Directory to save model artifacts.
        test_size: Fraction of data to use for testing.
        val_size: Fraction of training data to use for validation.
        random_state: Random state for reproducibility.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    pipeline = TrainingPipeline(
        data_path=data_path,
        target_column=target_column,
        model_type=model_type,
        output_dir=output_dir,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    return pipeline.run()
