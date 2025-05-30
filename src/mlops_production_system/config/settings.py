"""Configuration settings for the MLOps Production System."""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseSettings, Field

# Load environment variables from .env file
load_dotenv()

# Base directories
ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    LEVEL: str = Field(default="INFO")
    FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    FILE_PATH: Optional[Path] = Field(default=LOGS_DIR / "app.log")


class MLFlowSettings(BaseSettings):
    """MLflow configuration."""

    TRACKING_URI: str = Field(default=os.getenv("MLFLOW_TRACKING_URI", ""))
    EXPERIMENT_NAME: str = Field(default="mlops-production-system")
    MODEL_REGISTRY: str = Field(default=str(MODELS_DIR / "registry"))


class APISettings(BaseSettings):
    """API configuration."""

    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    DEBUG: bool = Field(default=False)
    RELOAD: bool = Field(default=False)
    WORKERS: int = Field(default=1)
    TIMEOUT: int = Field(default=60)
    API_V1_PREFIX: str = Field(default="/api/v1")


class DataSettings(BaseSettings):
    """Data configuration."""

    RAW_DATA_DIR: Path = Field(default=DATA_DIR / "raw")
    PROCESSED_DATA_DIR: Path = Field(default=DATA_DIR / "processed")
    FEATURES_DATA_DIR: Path = Field(default=DATA_DIR / "features")
    TRAIN_TEST_SPLIT_RATIO: float = Field(default=0.2)
    VALIDATION_SPLIT_RATIO: float = Field(default=0.1)
    RANDOM_STATE: int = Field(default=42)


class ModelSettings(BaseSettings):
    """Model configuration."""

    MODEL_NAME: str = Field(default="default_model")
    MODEL_VERSION: str = Field(default="0.1.0")
    MODEL_DIR: Path = Field(default=MODELS_DIR)
    HYPERPARAMS: Dict[str, Any] = Field(default={})
    EVALUATION_METRICS: list = Field(default=["accuracy", "f1", "precision", "recall"])


class MonitoringSettings(BaseSettings):
    """Monitoring configuration."""

    PROMETHEUS_PORT: int = Field(default=8001)
    GRAFANA_PORT: int = Field(default=3000)
    METRICS_PREFIX: str = Field(default="mlops_production_system")


class Settings(BaseSettings):
    """Main application settings."""

    ENV: str = Field(default="development")
    DEBUG: bool = Field(default=False)
    PROJECT_NAME: str = Field(default="MLOps Production System")
    VERSION: str = Field(default="0.1.0")
    
    # Sub-settings
    LOGGING: LoggingSettings = LoggingSettings()
    MLFLOW: MLFlowSettings = MLFlowSettings()
    API: APISettings = APISettings()
    DATA: DataSettings = DataSettings()
    MODEL: ModelSettings = ModelSettings()
    MONITORING: MonitoringSettings = MonitoringSettings()

    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"


# Create settings instance
settings = Settings()
