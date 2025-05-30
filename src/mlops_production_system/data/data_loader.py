"""Data loading and processing module."""

import logging
from pathlib import Path
from typing import Dict, Tuple, Union, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from mlops_production_system.config.settings import settings

logger = logging.getLogger(__name__)


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
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def load_data(self, file_name: str, **kwargs) -> pd.DataFrame:
        """Load data from file.
        
        Args:
            file_name: Name of the file to load.
            **kwargs: Additional arguments to pass to the loader.
        
        Returns:
            Loaded DataFrame.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        file_path = self.data_path / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        
        logger.info(f"Loading data from {file_path}")
        
        # Determine file format and load accordingly
        if file_path.suffix == ".csv":
            return pd.read_csv(file_path, **kwargs)
        elif file_path.suffix == ".parquet":
            return pd.read_parquet(file_path, **kwargs)
        elif file_path.suffix in [".xls", ".xlsx"]:
            return pd.read_excel(file_path, **kwargs)
        elif file_path.suffix == ".json":
            return pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def save_data(
        self, data: pd.DataFrame, file_name: str, output_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """Save data to file.
        
        Args:
            data: DataFrame to save.
            file_name: Name of the file to save.
            output_dir: Directory to save the file in.
                If None, uses the processed data directory.
        
        Returns:
            Path to the saved file.
        """
        output_dir = output_dir or settings.DATA.PROCESSED_DATA_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / file_name
        logger.info(f"Saving data to {file_path}")
        
        # Determine file format and save accordingly
        if file_path.suffix == ".csv":
            data.to_csv(file_path, index=False)
        elif file_path.suffix == ".parquet":
            data.to_parquet(file_path, index=False)
        elif file_path.suffix in [".xls", ".xlsx"]:
            data.to_excel(file_path, index=False)
        elif file_path.suffix == ".json":
            data.to_json(file_path, orient="records")
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        return file_path

    def split_data(
        self, 
        data: pd.DataFrame, 
        target_column: str,
        test_size: float = None,
        val_size: float = None,
        random_state: int = None,
        stratify: bool = False
    ) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """Split data into train, validation, and test sets.
        
        Args:
            data: DataFrame to split.
            target_column: Name of the target column.
            test_size: Size of the test set.
                If None, uses the default from settings.
            val_size: Size of the validation set.
                If None, uses the default from settings.
            random_state: Random state for reproducibility.
                If None, uses the default from settings.
            stratify: Whether to stratify the split based on the target.
        
        Returns:
            Dictionary with train, validation, and test data.
        """
        # Use default settings if not provided
        test_size = test_size or settings.DATA.TRAIN_TEST_SPLIT_RATIO
        val_size = val_size or settings.DATA.VALIDATION_SPLIT_RATIO
        random_state = random_state or settings.DATA.RANDOM_STATE
        
        # Extract features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split data into train and test
        stratify_data = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_data
        )
        
        # Further split train into train and validation
        if val_size > 0:
            # Adjust validation size relative to the training set
            adjusted_val_size = val_size / (1 - test_size)
            stratify_train = y_train if stratify else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, 
                test_size=adjusted_val_size,
                random_state=random_state,
                stratify=stratify_train
            )
            
            return {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
            }
        
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }


# Factory function for easy instantiation
def get_data_loader(data_path: Optional[Union[str, Path]] = None) -> DataLoader:
    """Get a DataLoader instance.
    
    Args:
        data_path: Path to the data file or directory.
    
    Returns:
        DataLoader instance.
    """
    return DataLoader(data_path=data_path)
