"""
Centralized configuration for Fraud Detection project.
All paths and parameters are defined here for easy management.
"""

from pathlib import Path
import os

# ============================================================================
# Project Root
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# ============================================================================
# Directory Paths
# ============================================================================

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw" / "ieee-fraud-detection"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_SYNTHETIC = DATA_DIR / "synthetic"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_XGB = MODELS_DIR / "xgb"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_ADVERSARIAL = OUTPUTS_DIR / "adversarial"
OUTPUTS_SUBMISSIONS = OUTPUTS_DIR / "submissions"
OUTPUTS_PLOTS = OUTPUTS_DIR / "plots"

# TabDiff directory
TABDIFF_DIR = PROJECT_ROOT / "TabDiff"

# ============================================================================
# File Paths
# ============================================================================

# Input data files
TRAIN_IDENTITY = DATA_RAW / "train_identity.csv"
TRAIN_TRANSACTION = DATA_RAW / "train_transaction.csv"
TEST_IDENTITY = DATA_RAW / "test_identity.csv"
TEST_TRANSACTION = DATA_RAW / "test_transaction.csv"

# Synthetic data
SYNTHETIC_FRAUD = DATA_SYNTHETIC / "tabdiff_synthetic_fraud_pca_1000.csv"
TABDIFF_SAMPLES = (
    TABDIFF_DIR / "tabdiff" / "result" / "fraud_data" / "quick_fraud" / "1" / "samples.csv"
)

# Adversarial samples
ADVERSARIAL_RAW = OUTPUTS_ADVERSARIAL / "adversarial_samples.csv"
ADVERSARIAL_PURIFIED = OUTPUTS_ADVERSARIAL / "adversarial_samples_purified.csv"
ADVERSARIAL_CLEAN = OUTPUTS_ADVERSARIAL / "adversarial_samples_purified_clean.csv"

# Submissions
SUBMISSION_BASE = OUTPUTS_SUBMISSIONS / "submission.csv"
SUBMISSION_NEW = OUTPUTS_SUBMISSIONS / "new_submission.csv"
SUBMISSION_MODEL1 = OUTPUTS_SUBMISSIONS / "new_submission_model1.csv"
SUBMISSION_MODEL2 = OUTPUTS_SUBMISSIONS / "new_submission_model2.csv"
SUBMISSION_MODEL3 = OUTPUTS_SUBMISSIONS / "new_submission_model3.csv"

# ============================================================================
# Model Parameters
# ============================================================================

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "use_label_encoder": False,
    "learning_rate": 0.05,
    "n_estimators": 5000,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 10,
}

# ============================================================================
# TabDiff Parameters
# ============================================================================

TABDIFF_CONFIG = {
    "dataname": "fraud_data",
    "exp_name": "quick_fraud",
    "steps": 500,
    "checkpoint_freq": 250,
    "batch_size": 512,
    "sample_batch_size": 256,
    "num_timesteps": 5,
    "synthetic_samples": 1000,
}

# Noise schedule parameters for adversarial generation
NOISE_PARAMS = {
    "sigma_min": 0.002,
    "sigma_max": 80,
    "rho": 7,
}

# ============================================================================
# Preprocessing Parameters
# ============================================================================

PREPROCESSING = {
    "outlier_threshold": 3,  # z-score threshold
    "missing_threshold": 0.90,  # drop columns with >90% missing
    "variance_threshold": 0.90,  # PCA: retain 90% variance
    "test_size": 0.2,  # train/validation split
    "random_state": 42,
}

# ============================================================================
# Adversarial Generation Parameters
# ============================================================================

ADVERSARIAL_PARAMS = {
    "fraud_sample_ratio": 0.2,  # 20% of fraud cases
    "noise_steps": 100,  # Number of noise steps
    "purification_t": 0.1,  # Timestep for purification
    "purification_steps": 5,  # Reverse diffusion steps
    "adversarial_weight": 3.0,  # Sample weight for adversarial data
}

# ============================================================================
# Utility Functions
# ============================================================================

def ensure_directories():
    """Create all necessary directories if they don't exist."""
    dirs = [
        DATA_DIR,
        DATA_RAW,
        DATA_PROCESSED,
        DATA_SYNTHETIC,
        MODELS_DIR,
        MODELS_XGB,
        OUTPUTS_DIR,
        OUTPUTS_ADVERSARIAL,
        OUTPUTS_SUBMISSIONS,
        OUTPUTS_PLOTS,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_latest_xgb_model():
    """Find the most recent XGBoost model in the models directory."""
    if not MODELS_XGB.exists():
        return None
    files = list(MODELS_XGB.glob("*.json"))
    if not files:
        return None
    return max(files, key=os.path.getctime)


def get_tabdiff_checkpoint():
    """Find the latest TabDiff checkpoint."""
    ckpt_dir = TABDIFF_DIR / "tabdiff" / "ckpt" / "fraud_data" / "quick_fraud"

    # Try alternative paths
    if not ckpt_dir.exists():
        alt_paths = [
            TABDIFF_DIR / "tabdiff" / "ckpt" / "fraud_data" / "learnable_schedule",
            TABDIFF_DIR / "tabdiff" / "ckpt" / "fraud_data",
        ]
        for alt in alt_paths:
            if alt.exists():
                ckpt_dir = alt
                break

    if not ckpt_dir.exists():
        return None

    # Find best checkpoint
    best_ema = list(ckpt_dir.glob("best_ema_model_*.pt"))
    best_model = list(ckpt_dir.glob("best_model_*.pt"))
    models = list(ckpt_dir.glob("model_*.pt"))

    if best_ema:
        return max(best_ema, key=lambda x: int(x.stem.split("_")[-1]))
    elif best_model:
        return max(best_model, key=lambda x: int(x.stem.split("_")[-1]))
    elif models:
        return max(models, key=lambda x: int(x.stem.split("_")[-1]))

    return None
