"""Data loading and processing module."""

import logging
from pathlib import Path
from typing import Dict, Tuple, Union, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from mlops_forge.config.settings import settings
from mlops_forge.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Data loading and processing class."""

    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        """Initialize DataLoader.
        
        Args:
            data_path: Path to the data file or directory.
                If None, uses the default raw data directory.
        """
        self.data_path = data_path or settings.DATA.RAW_DATA_DIR
        self.data_path = Path(self.data_path)
        
        # Ensure data directories exist
        for dir_path in [
            settings.DATA.RAW_DATA_DIR,
            settings.DATA.PROCESSED_DATA_DIR,
            settings.DATA.FEATURES_DATA_DIR,
        ]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    def load_data(self, file_name: Optional[str] = None) -> pd.DataFrame:
        """Load data from file.
        
        Args:
            file_name: Name of the file to load. If None, uses the data_path directly.
        
        Returns:
            DataFrame containing the data.
        
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is not supported.
        """
        file_path = self.data_path / file_name if file_name else self.data_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Handle different file formats
        if file_path.suffix == ".csv":
            logger.info(f"Loading CSV data from {file_path}")
            return pd.read_csv(file_path)
        elif file_path.suffix == ".parquet":
            logger.info(f"Loading Parquet data from {file_path}")
            return pd.read_parquet(file_path)
        elif file_path.suffix == ".json":
            logger.info(f"Loading JSON data from {file_path}")
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data.
        
        Args:
            data: DataFrame to preprocess.
        
        Returns:
            Preprocessed DataFrame.
        """
        logger.info("Preprocessing data")
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Handle missing values (example: fill numeric with mean, categorical with mode)
        for col in data.columns:
            if data[col].isna().any():
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col].fillna(data[col].mean())
                else:
                    data[col] = data[col].fillna(data[col].mode()[0])
        
        return data
    
    def save_data(self, data: pd.DataFrame, file_name: str, output_dir: Optional[Union[str, Path]] = None) -> Path:
        """Save data to file.
        
        Args:
            data: DataFrame to save.
            file_name: Name of the output file.
            output_dir: Directory to save the file. If None, uses processed data directory.
            
        Returns:
            Path to the saved file.
            
        Raises:
            ValueError: If the file format is not supported.
        """
        output_dir = output_dir or settings.DATA.PROCESSED_DATA_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        file_path = output_dir / file_name
        
        # Handle different file formats
        if file_path.suffix == ".csv":
            logger.info(f"Saving CSV data to {file_path}")
            data.to_csv(file_path, index=False)
        elif file_path.suffix == ".parquet":
            logger.info(f"Saving Parquet data to {file_path}")
            data.to_parquet(file_path, index=False)
        elif file_path.suffix == ".json":
            logger.info(f"Saving JSON data to {file_path}")
            data.to_json(file_path, orient="records")
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        return file_path
    
    def split_data(
        self, 
        data: pd.DataFrame, 
        target_column: str,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """Split data into train, validation, and test sets.
        
        Args:
            data: DataFrame to split.
            target_column: Name of the target column.
            test_size: Size of the test set (proportion of the total).
                If None, uses settings.DATA.TRAIN_TEST_SPLIT_RATIO.
            val_size: Size of the validation set (proportion of the train set).
                If None, uses settings.DATA.VALIDATION_SPLIT_RATIO.
                If 0, no validation set is created.
            random_state: Random state for reproducibility.
                If None, uses settings.DATA.RANDOM_STATE.
        
        Returns:
            Dictionary containing the split data:
                - X_train: Features for training
                - y_train: Targets for training
                - X_val: Features for validation (only if val_size > 0)
                - y_val: Targets for validation (only if val_size > 0)
                - X_test: Features for testing
                - y_test: Targets for testing
        """
        logger.info("Splitting data into train, validation, and test sets")
        
        # Use default values from settings if not provided
        test_size = test_size or settings.DATA.TRAIN_TEST_SPLIT_RATIO
        val_size = val_size if val_size is not None else settings.DATA.VALIDATION_SPLIT_RATIO
        random_state = random_state or settings.DATA.RANDOM_STATE
        
        # Extract features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # First split: train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        result = {
            "X_test": X_test,
            "y_test": y_test,
        }
        
        # Second split: train and val (only if val_size > 0)
        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size, random_state=random_state
            )
            
            result.update({
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
            })
        else:
            # No validation split needed
            result.update({
                "X_train": X_train_val,
                "y_train": y_train_val,
            })
        
        return result
    
    def save_processed_data(
        self, 
        data: pd.DataFrame,
        file_name: str,
        output_dir: Optional[Path] = None
    ):
        """Save processed data to file.
        
        Args:
            data: DataFrame to save.
            file_name: Name of the output file.
            output_dir: Directory to save the file.
                If None, uses settings.DATA.PROCESSED_DATA_DIR.
        """
        output_dir = output_dir or settings.DATA.PROCESSED_DATA_DIR
        output_path = output_dir / file_name
        
        # Create directory if it doesn't exist
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save data in appropriate format based on extension
        if output_path.suffix == ".csv":
            logger.info(f"Saving processed data to CSV: {output_path}")
            data.to_csv(output_path, index=False)
        elif output_path.suffix == ".parquet":
            logger.info(f"Saving processed data to Parquet: {output_path}")
            data.to_parquet(output_path, index=False)
        elif output_path.suffix == ".json":
            logger.info(f"Saving processed data to JSON: {output_path}")
            data.to_json(output_path, orient="records")
        else:
            # Default to Parquet
            output_path = output_path.with_suffix(".parquet")
            logger.info(f"Saving processed data to Parquet: {output_path}")
            data.to_parquet(output_path, index=False)


# Factory function for easy instantiation
def get_data_loader(data_path: Optional[Union[str, Path]] = None) -> DataLoader:
    """Get a DataLoader instance.
    
    Args:
        data_path: Path to the data file or directory.
    
    Returns:
        DataLoader instance.
    """
    return DataLoader(data_path)
