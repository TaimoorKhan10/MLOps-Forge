"""Script to generate sample dataset for demonstration."""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from pathlib import Path

def generate_classification_data(n_samples=1000, n_features=10, n_informative=5, 
                                n_redundant=2, n_classes=2, random_state=42):
    """Generate classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def generate_regression_data(n_samples=1000, n_features=10, n_informative=5, 
                            noise=0.1, random_state=42):
    """Generate regression dataset."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def add_categorical_features(df, n_categorical=2, n_categories=3, random_state=42):
    """Add categorical features to DataFrame."""
    np.random.seed(random_state)
    
    for i in range(n_categorical):
        categories = [f'category_{j}' for j in range(n_categories)]
        df[f'categorical_{i+1}'] = np.random.choice(categories, size=len(df))
    
    return df

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate sample dataset')
    parser.add_argument('--type', type=str, default='classification', 
                        choices=['classification', 'regression'],
                        help='Type of dataset to generate')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples')
    parser.add_argument('--features', type=int, default=10,
                        help='Number of features')
    parser.add_argument('--categorical', type=int, default=2,
                        help='Number of categorical features')
    parser.add_argument('--output', type=str, default='data/raw/sample_data.csv',
                        help='Output file path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Generate data
    if args.type == 'classification':
        df = generate_classification_data(
            n_samples=args.samples,
            n_features=args.features,
            random_state=args.seed
        )
    else:
        df = generate_regression_data(
            n_samples=args.samples,
            n_features=args.features,
            random_state=args.seed
        )
    
    # Add categorical features
    df = add_categorical_features(
        df=df, 
        n_categorical=args.categorical,
        random_state=args.seed
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"Sample dataset generated: {args.output}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"First 5 rows:")
    print(df.head())

if __name__ == '__main__':
    main()
