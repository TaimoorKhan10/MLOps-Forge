"""Unit tests for data loader."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from mlops_forge.data.data_loader import DataLoader, get_data_loader


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5],
        "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5],
        "target": [0, 1, 0, 1, 1]
    })


@pytest.fixture
def temp_csv_file(sample_data):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        sample_data.to_csv(temp_file.name, index=False)
        temp_file_path = temp_file.name
    
    yield temp_file_path
    
    # Clean up
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)


class TestDataLoader:
    """Test data loader class."""
    
    def test_init(self):
        """Test initialization."""
        data_loader = DataLoader()
        assert isinstance(data_loader.data_path, Path)
    
    def test_load_data_csv(self, temp_csv_file):
        """Test loading data from CSV file."""
        data_loader = DataLoader(data_path=os.path.dirname(temp_csv_file))
        data = data_loader.load_data(os.path.basename(temp_csv_file))
        
        assert isinstance(data, pd.DataFrame)
        assert data.shape == (5, 3)
        assert list(data.columns) == ["feature_1", "feature_2", "target"]
    
    def test_load_data_file_not_found(self):
        """Test loading data from non-existent file."""
        data_loader = DataLoader()
        
        with pytest.raises(FileNotFoundError):
            data_loader.load_data("non_existent_file.csv")
    
    def test_load_data_unsupported_format(self, temp_csv_file):
        """Test loading data from unsupported file format."""
        # Rename file to unsupported extension
        unsupported_file = temp_csv_file.replace(".csv", ".xyz")
        os.rename(temp_csv_file, unsupported_file)
        
        try:
            data_loader = DataLoader(data_path=os.path.dirname(unsupported_file))
            
            with pytest.raises(ValueError, match="Unsupported file format"):
                data_loader.load_data(os.path.basename(unsupported_file))
        finally:
            # Clean up
            if os.path.exists(unsupported_file):
                os.remove(unsupported_file)
    
    def test_save_data(self, sample_data):
        """Test saving data to file."""
        data_loader = DataLoader()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = data_loader.save_data(
                data=sample_data,
                file_name="test_data.csv",
                output_dir=temp_dir
            )
            
            assert os.path.exists(file_path)
            
            # Load the saved data and verify
            loaded_data = pd.read_csv(file_path)
            pd.testing.assert_frame_equal(loaded_data, sample_data)
    
    def test_split_data(self, sample_data):
        """Test splitting data."""
        data_loader = DataLoader()
        
        # Test with validation set
        splits = data_loader.split_data(
            data=sample_data,
            target_column="target",
            test_size=0.2,
            val_size=0.25,
            random_state=42
        )
        
        assert "X_train" in splits
        assert "y_train" in splits
        assert "X_val" in splits
        assert "y_val" in splits
        assert "X_test" in splits
        assert "y_test" in splits
        
        # Test without validation set
        splits = data_loader.split_data(
            data=sample_data,
            target_column="target",
            test_size=0.2,
            val_size=0,
            random_state=42
        )
        
        assert "X_train" in splits
        assert "y_train" in splits
        assert "X_test" in splits
        assert "y_test" in splits
        assert "X_val" not in splits
        assert "y_val" not in splits
    
    def test_get_data_loader(self):
        """Test get_data_loader factory function."""
        data_loader = get_data_loader()
        assert isinstance(data_loader, DataLoader)
