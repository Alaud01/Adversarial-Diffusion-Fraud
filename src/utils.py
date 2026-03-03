"""
Utility functions used across the fraud detection pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os
from typing import List, Optional


def get_latest_model(model_dir: str = "models/xgb") -> Optional[str]:
    """Find the most recent XGBoost model file."""
    model_path = Path(model_dir)
    if not model_path.exists():
        return None

    files = list(model_path.glob("*.json"))
    if not files:
        return None

    return str(max(files, key=os.path.getctime))


def get_tabdiff_noise_sigma(t: float, sigma_min: float = 0.002, sigma_max: float = 80, rho: float = 7) -> float:
    """Calculate noise standard deviation for a given diffusion timestep."""
    sigma = (sigma_min ** (1 / rho) + t * (sigma_max ** (1 / rho) - sigma_min ** (1 / rho))) ** rho
    return sigma


def align_features(X: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    """Align dataframe columns with expected features."""
    for col in expected_features:
        if col not in X.columns:
            X[col] = np.nan

    X = X[expected_features]
    return X


def validate_dataframe(df: pd.DataFrame, required_cols: List[str] = None) -> bool:
    """Validate dataframe has expected structure."""
    if df is None or df.empty:
        raise ValueError("DataFrame is empty")

    if required_cols:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    return True


def compute_class_weights(y: pd.Series) -> float:
    """Compute scale_pos_weight for imbalanced classification."""
    n_negative = (y == 0).sum()
    n_positive = (y == 1).sum()

    if n_positive == 0:
        return 1.0

    return n_negative / n_positive


def split_train_val(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Split data into train and validation sets with stratification."""
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def save_metrics(metrics: dict, output_path: str):
    """Save metrics dictionary to JSON file."""
    import json

    metrics_clean = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer, np.floating)):
            metrics_clean[k] = float(v)
        elif isinstance(v, np.ndarray):
            metrics_clean[k] = v.tolist()
        else:
            metrics_clean[k] = v

    with open(output_path, "w") as f:
        json.dump(metrics_clean, f, indent=4)

    print(f"Saved metrics to {output_path}")


def load_data_safe(path: str, **kwargs) -> pd.DataFrame:
    """Safely load CSV with error handling."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    try:
        df = pd.read_csv(path, **kwargs)
        print(f"Loaded {len(df)} rows from {path}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading {path}: {e}")


def generate_submission(predictions: np.ndarray, transaction_ids: pd.Series, output_path: str):
    """Generate and save submission file in Kaggle format."""
    submission = pd.DataFrame({
        "TransactionID": transaction_ids,
        "isFraud": predictions,
    })

    submission.to_csv(output_path, index=False)
    print(f"Saved submission ({len(submission)} rows) to {output_path}")


def check_gpu_available() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device(prefer_gpu: bool = True) -> str:
    """Get appropriate device (cuda or cpu)."""
    if prefer_gpu and check_gpu_available():
        return "cuda"
    return "cpu"
