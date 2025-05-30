"""Distributed training module for MLOps-Forge."""

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

from mlops_forge.config.settings import settings
from mlops_forge.data.data_loader import get_data_loader
from mlops_forge.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TabularDataset(Dataset):
    """Dataset for tabular data."""

    def __init__(self, features, targets):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
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
        x = self.layer1(x)
        x = self.relu(x)
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.relu(x)
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
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def load_data(dataset_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data for training.
    
    Args:
        dataset_path: Path to the dataset
        
    Returns:
        Tuple of features and targets
    """
    # In a real scenario, we'd use a more sophisticated data loading approach
    data = pd.read_parquet(dataset_path)
    
    # Assume the last column is the target
    features = data.iloc[:, :-1]
    targets = data.iloc[:, -1]
    
    return features, targets


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
    # Initialize distributed environment
    setup(rank, world_size)
    
    # Set up logging for this process
    logger.info(f"Starting training process on rank {rank}")
    
    # Initialize MLflow if this is the master process
    if is_master:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        run_id = mlflow.start_run(run_name=f"distributed_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Log parameters
        mlflow.log_params({
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "world_size": world_size
        })
    
    try:
        _train_process(
            rank=rank,
            world_size=world_size,
            dataset_path=dataset_path,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            hidden_dim=hidden_dim,
            is_master=is_master
        )
    finally:
        # Clean up
        cleanup()
        if is_master:
            mlflow.end_run()


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
    # Load data
    features, targets = load_data(dataset_path)
    
    # Create dataset and dataloader
    dataset = TabularDataset(features, targets)
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    input_dim = features.shape[1]
    num_classes = len(targets.unique())
    model = SimpleModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes)
    
    # Move model to device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 10 == 9:
                logger.info(f"Rank {rank}, Epoch {epoch+1}, Batch {i+1}, "
                          f"Loss: {running_loss/10:.3f}, "
                          f"Acc: {100.*correct/total:.2f}%")
                running_loss = 0.0
        
        # Calculate epoch metrics
        epoch_acc = 100. * correct / total
        epoch_loss = running_loss / len(dataloader)
        
        # Log metrics for master process
        if is_master:
            mlflow.log_metrics({
                "epoch": epoch+1,
                "accuracy": epoch_acc,
                "loss": epoch_loss
            }, step=epoch)
            
            # Save model checkpoint
            if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'accuracy': epoch_acc
                }
                torch.save(checkpoint, f"model_checkpoint_epoch_{epoch+1}.pt")
                mlflow.log_artifact(f"model_checkpoint_epoch_{epoch+1}.pt")
    
    # Final synchronization
    dist.barrier()
    
    if is_master:
        logger.info("Training completed successfully")


def main():
    """Main entry point for the distributed trainer."""
    parser = argparse.ArgumentParser(description="Distributed Training Script")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="GPUs per node")
    
    args = parser.parse_args()
    
    world_size = args.num_nodes * args.gpus_per_node
    
    if world_size == 1:
        # Single GPU training
        train(
            rank=0,
            world_size=1,
            dataset_path=args.data_path,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            is_master=True
        )
    else:
        # Multi-GPU training
        mp.spawn(
            train,
            args=(
                world_size,
                args.data_path,
                args.batch_size,
                args.learning_rate,
                args.epochs,
                args.hidden_dim,
                True
            ),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    main()
