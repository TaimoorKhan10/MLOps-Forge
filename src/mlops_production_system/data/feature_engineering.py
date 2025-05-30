"""Feature engineering and preprocessing module."""

import logging
from typing import Dict, List, Optional, Union, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from mlops_production_system.config.settings import settings

logger = logging.getLogger(__name__)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selector transformer."""

    def __init__(self, feature_names: List[str]):
        """Initialize feature selector.
        
        Args:
            feature_names: List of feature names to select.
        """
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        """Fit transformer."""
        return self
        
    def transform(self, X):
        """Transform the data by selecting features."""
        return X[self.feature_names]


class CustomTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering."""
    
    def __init__(self, transform_fn: Callable):
        """Initialize custom transformer.
        
        Args:
            transform_fn: Function to apply to the data.
        """
        self.transform_fn = transform_fn
        
    def fit(self, X, y=None):
        """Fit transformer."""
        return self
        
    def transform(self, X):
        """Transform the data by applying the transform function."""
        return self.transform_fn(X)


class FeatureEngineer:
    """Feature engineering class."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.preprocessing_pipeline = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        
    def fit_preprocessing_pipeline(
        self,
        data: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        custom_transformers: Optional[Dict[str, Pipeline]] = None
    ) -> "FeatureEngineer":
        """Fit preprocessing pipeline.
        
        Args:
            data: Input DataFrame.
            categorical_features: List of categorical feature names.
            numerical_features: List of numerical feature names.
            custom_transformers: Dictionary of custom transformers.
        
        Returns:
            Self for method chaining.
        """
        # Infer feature types if not specified
        if categorical_features is None:
            categorical_features = data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            
        if numerical_features is None:
            numerical_features = data.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.feature_names = categorical_features + numerical_features
        
        logger.info(f"Categorical features: {categorical_features}")
        logger.info(f"Numerical features: {numerical_features}")
        
        # Create preprocessing pipelines
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        # Create column transformer
        transformers = [
            ("categorical", categorical_pipeline, categorical_features),
            ("numerical", numerical_pipeline, numerical_features)
        ]
        
        # Add custom transformers if provided
        if custom_transformers:
            for name, (transformer, columns) in custom_transformers.items():
                transformers.append((name, transformer, columns))
        
        # Create preprocessing pipeline
        self.preprocessing_pipeline = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )
        
        # Fit preprocessing pipeline
        self.preprocessing_pipeline.fit(data)
        
        return self
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform data using the preprocessing pipeline.
        
        Args:
            data: Input DataFrame.
        
        Returns:
            Transformed data.
        
        Raises:
            ValueError: If preprocessing pipeline is not fitted.
        """
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessing pipeline is not fitted. Call fit_preprocessing_pipeline first.")
            
        return self.preprocessing_pipeline.transform(data)
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        custom_transformers: Optional[Dict[str, Pipeline]] = None
    ) -> np.ndarray:
        """Fit and transform data.
        
        Args:
            data: Input DataFrame.
            categorical_features: List of categorical feature names.
            numerical_features: List of numerical feature names.
            custom_transformers: Dictionary of custom transformers.
        
        Returns:
            Transformed data.
        """
        self.fit_preprocessing_pipeline(
            data=data,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            custom_transformers=custom_transformers
        )
        
        return self.transform(data)
    
    def save_features(self, output_path: Optional[str] = None) -> str:
        """Save feature metadata.
        
        Args:
            output_path: Path to save feature metadata.
        
        Returns:
            Path to the saved feature metadata.
        
        Raises:
            ValueError: If preprocessing pipeline is not fitted.
        """
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessing pipeline is not fitted. Call fit_preprocessing_pipeline first.")
            
        # Determine output path
        output_path = output_path or f"{settings.DATA.FEATURES_DATA_DIR}/feature_metadata.json"
        
        # Create feature metadata
        feature_metadata = {
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "feature_names": self.feature_names,
        }
        
        # Save feature metadata
        import json
        with open(output_path, "w") as f:
            json.dump(feature_metadata, f, indent=2)
            
        logger.info(f"Feature metadata saved to {output_path}")
        
        return output_path


# Factory function for easy instantiation
def get_feature_engineer() -> FeatureEngineer:
    """Get a FeatureEngineer instance.
    
    Returns:
        FeatureEngineer instance.
    """
    return FeatureEngineer()
