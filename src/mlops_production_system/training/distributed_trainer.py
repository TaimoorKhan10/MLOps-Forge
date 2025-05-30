"""Distributed training module for MLOps Production System."""

import os
import argparse
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import pandas as pd
import numpy as np
import mlflow

from mlops_production_system.config.settings import settings
from mlops_production_system.data.data_loader import get_data_loader
from mlops_production_system.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TabularDataset(Dataset):
    """Dataset for tabular data."""

    def __init__(self, features, targets):
        """Initialize the dataset.
        
        Args:
            features: Feature matrix
            targets: Target vector
        """
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.long)

    def __len__(self):
        """Get the number of samples."""
        return len(self.features)

    def __getitem__(self, idx):
        """Get a sample."""
        return self.features[idx], self.targets[idx]


class SimpleModel(nn.Module):
    """Simple neural network model for tabular data."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 2):
        """Initialize the model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            output_dim: Number of output classes
        """
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.relu(self.layer1(x))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x


def setup(rank, world_size):
    """Initialize the distributed environment.
    
    Args:
        rank: Rank of the current process
        world_size: Number of processes
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    logger.info(f"Initialized process group: rank={rank}, world_size={world_size}")


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()
    logger.info("Destroyed process group")


def load_data(dataset_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data for training.
    
    Args:
        dataset_path: Path to the dataset
        
    Returns:
        Tuple of features and targets
    """
    # Load the data
    df = pd.read_csv(dataset_path)
    
    # Extract features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    return X, y


def train(
    rank: int,
    world_size: int,
    dataset_path: str,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    epochs: int = 10,
    hidden_dim: int = 128,
    is_master: bool = False
):
    """Train the model in a distributed setting.
    
    Args:
        rank: Rank of the current process
        world_size: Number of processes
        dataset_path: Path to the dataset
        batch_size: Batch size
        learning_rate: Learning rate
        epochs: Number of epochs
        hidden_dim: Hidden dimension size
        is_master: Whether this process is the master
    """
    # Set up distributed training
    setup(rank, world_size)
    
    # Configure MLflow if this is the master process
    if is_master:
        mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', settings.MLFLOW.TRACKING_URI))
        mlflow.set_experiment(os.environ.get('MLFLOW_EXPERIMENT_NAME', settings.MLFLOW.EXPERIMENT_NAME))
        
        # Start a new run
        with mlflow.start_run(run_name=f"distributed_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params({
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "hidden_dim": hidden_dim,
                "world_size": world_size,
                "distributed": True
            })
            
            # Log the dataset path
            mlflow.log_param("dataset_path", dataset_path)
            
            logger.info(f"Started MLflow run with params: batch_size={batch_size}, lr={learning_rate}, epochs={epochs}")
            
            # Perform the actual training
            _train_process(rank, world_size, dataset_path, batch_size, learning_rate, epochs, hidden_dim, is_master)
    else:
        # Non-master processes just do the training without MLflow
        _train_process(rank, world_size, dataset_path, batch_size, learning_rate, epochs, hidden_dim, is_master)
    
    # Clean up
    cleanup()


def _train_process(
    rank: int,
    world_size: int,
    dataset_path: str,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    epochs: int = 10,
    hidden_dim: int = 128,
    is_master: bool = False
):
    """Internal function to handle the training process.
    
    Args:
        rank: Rank of the current process
        world_size: Number of processes
        dataset_path: Path to the dataset
        batch_size: Batch size
        learning_rate: Learning rate
        epochs: Number of epochs
        hidden_dim: Hidden dimension size
        is_master: Whether this process is the master
    """
    # Load and prepare data
    X, y = load_data(dataset_path)
    
    # Create dataset and sampler
    dataset = TabularDataset(X, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    # Create model
    input_dim = X.shape[1]
    output_dim = len(y.unique())
    model = SimpleModel(input_dim, hidden_dim, output_dim)
    
    # Move model to GPU if available
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank % torch.cuda.device_count()] if torch.cuda.is_available() else None)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch)  # Required for proper shuffling
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Rank {rank} | Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | " +
                           f"Loss: {running_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%")
        
        # Gather metrics from all processes
        epoch_loss = torch.tensor([running_loss / len(dataloader)]).to(device)
        epoch_accuracy = torch.tensor([correct / total]).to(device)
        
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_accuracy, op=dist.ReduceOp.SUM)
        
        epoch_loss = epoch_loss.item() / world_size
        epoch_accuracy = epoch_accuracy.item() / world_size
        
        # Log metrics for master process
        if is_master:
            logger.info(f"Epoch {epoch+1}/{epochs} | Average Loss: {epoch_loss:.3f} | Average Acc: {100.*epoch_accuracy:.3f}%")
            
            mlflow.log_metrics({
                "train_loss": epoch_loss,
                "train_accuracy": epoch_accuracy
            }, step=epoch)
    
    # Save model (only master process)
    if is_master:
        # Save model artifacts
        model_path = os.path.join(settings.MODEL.MODEL_DIR, f"distributed_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        torch.save(model.module.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model.module, "model")
        
        # Save model metadata
        model_metadata = {
            "model_name": "distributed_nn",
            "model_version": settings.MODEL.VERSION,
            "model_type": "neural_network",
            "created_at": datetime.now().isoformat(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "distributed_training": True,
            "world_size": world_size,
            "metrics": {
                "train_loss": epoch_loss,
                "train_accuracy": epoch_accuracy
            }
        }
        
        with open(os.path.join(settings.MODEL.MODEL_DIR, "distributed_model_metadata.json"), "w") as f:
            json.dump(model_metadata, f, indent=2)


def main():
    """Main entry point for the distributed trainer."""
    parser = argparse.ArgumentParser(description="Distributed Training")
    parser.add_argument("--master", action="store_true", help="Run as master node")
    parser.add_argument("--worker", action="store_true", help="Run as worker node")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker nodes")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension size")
    
    args = parser.parse_args()
    
    # Get rank from environment or command line
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", str(args.num_workers)))
    
    logger.info(f"Starting distributed training with rank={rank}, world_size={world_size}")
    
    # Run the training
    train(
        rank=rank,
        world_size=world_size,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        is_master=args.master
    )


if __name__ == "__main__":
    main()
