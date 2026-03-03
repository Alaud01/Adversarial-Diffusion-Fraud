# Fraud Detection with Synthetic Data Generation

A comprehensive fraud detection pipeline using XGBoost, enhanced with synthetic data generation via TabDiff diffusion models and adversarial sample generation for robust model training.

## 📁 Project Structure

```
Fraud-Detection-Diffusion/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.py                 # Centralized configuration and paths
│
├── notebooks/                # Jupyter notebooks
│   └── detect_fraud.ipynb   # Main analysis and baseline model
│
├── src/                      # Source code modules (shared utilities)
│   ├── __init__.py          # Package initialization
│   ├── preprocessing.py     # Data preprocessing classes
│   ├── models.py            # Model training and evaluation
│   ├── utils.py             # Helper functions
│   └── visualization.py     # Plotting utilities
│
├── scripts/                  # Executable scripts
│   ├── generate_tabdiff_fraud.py      # Generate synthetic fraud with TabDiff
│   ├── generate_adversarial_fraud.py  # Generate adversarial samples
│   └── retrain_with_synthetic.py      # Retrain models with synthetic data
│
├── data/                     # Data directory
│   ├── raw/                 # Raw input data
│   ├── processed/           # Processed datasets
│   └── synthetic/           # Synthetic generated data
│
├── models/                   # Saved models
│   └── xgb/                 # XGBoost model checkpoints
│
├── outputs/                  # Generated outputs
│   ├── adversarial/         # Adversarial sample CSVs
│   ├── submissions/         # Kaggle submission files
│   └── plots/               # Visualization plots
│
└── TabDiff/                  # TabDiff submodule (synthetic data generation)
```

## 🚀 Quick Start

### Prerequisites

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Place the IEEE Fraud Detection dataset in `data/raw/`:
```
data/raw/
└── ieee-fraud-detection/
    ├── train_identity.csv
    ├── train_transaction.csv
    ├── test_identity.csv
    └── test_transaction.csv
```

## 📊 Workflow Overview

The project follows a 3-stage pipeline:

### Stage 1: Baseline Model (`notebooks/detect_fraud.ipynb`)
- **Purpose**: Establish baseline fraud detection performance
- **Input**: Raw IEEE Fraud Detection dataset
- **Output**: Trained XGBoost model with ~96% ROC-AUC
- **Key Steps**:
  1. Data merging and preprocessing
  2. Feature engineering (email domains, device info, temporal features)
  3. Outlier removal (>3σ in TransactionAmt)
  4. Encoding and scaling (StandardScaler, MinMaxScaler, LabelEncoder)
  5. PCA dimensionality reduction (V-columns → PC columns)
  6. XGBoost model training with class balancing

### Stage 2: Synthetic Data Generation (`scripts/generate_tabdiff_fraud.py`)
- **Purpose**: Generate additional fraud samples using diffusion models
- **Input**: Fraud cases from processed dataset
- **Output**: 1000 synthetic fraud samples in PCA space
- **Key Steps**:
  1. Apply full preprocessing pipeline (same as Stage 1)
  2. Extract fraud cases after PCA transformation
  3. Train TabDiff diffusion model on processed fraud samples
  4. Generate synthetic samples using reverse diffusion
  5. Save to `data/synthetic/tabdiff_synthetic_fraud_pca_1000.csv`

### Stage 3: Adversarial Sample Generation (`scripts/generate_adversarial_fraud.py`)
- **Purpose**: Create adversarial examples to improve model robustness
- **Input**: Validation set fraud cases + trained XGBoost model
- **Output**: Purified adversarial samples
- **Key Steps**:
  1. Select 10-20% of validation fraud cases
  2. **Attack Loop**: Iteratively add TabDiff-style noise until model predicts "Safe"
     - Uses PowerMean noise schedule (σ: 0.002 → 80)
     - Gaussian noise for numerical features
  3. **Purification**: Apply TabDiff reverse diffusion to make samples realistic
  4. Save to `outputs/adversarial/adversarial_samples_purified_clean.csv`

### Stage 4: Model Retraining (`scripts/retrain_with_synthetic.py`)
- **Purpose**: Retrain models with synthetic and adversarial data
- **Input**: Original + synthetic + adversarial data
- **Output**: 3 trained models with performance comparison
- **Key Steps**:
  1. Train 3 models:
     - **Model 1**: Original data only (baseline)
     - **Model 2**: Original + Synthetic fraud
     - **Model 3**: Original + Synthetic + Adversarial (3× weight on adversarial)
  2. Evaluate on validation set
  3. Generate predictions for test set
  4. Save submissions to `outputs/submissions/`

## 🔧 Configuration

All paths and parameters are centralized in `src/config.py`:

```python
# Key paths
DATA_RAW = Path("data/raw/ieee-fraud-detection")
DATA_PROCESSED = Path("data/processed")
DATA_SYNTHETIC = Path("data/synthetic")
MODELS_DIR = Path("models/xgb")
OUTPUTS_ADVERSARIAL = Path("outputs/adversarial")
OUTPUTS_SUBMISSIONS = Path("outputs/submissions")
OUTPUTS_PLOTS = Path("outputs/plots")

# Model parameters
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'n_estimators': 5000,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'early_stopping_rounds': 10
}

# TabDiff parameters
TABDIFF_STEPS = 500
TABDIFF_CHECKPOINT_FREQ = 250
SYNTHETIC_SAMPLES = 1000
```

## 📖 Usage Examples

### Run Full Pipeline

```bash
# 1. Start with baseline (run notebook)
jupyter notebook notebooks/detect_fraud.ipynb

# 2. Generate synthetic fraud samples
python scripts/generate_tabdiff_fraud.py

# 3. Generate adversarial samples
python scripts/generate_adversarial_fraud.py

# 4. Retrain with all data
python scripts/retrain_with_synthetic.py
```

### Run Individual Scripts

```bash
# Generate synthetic data only
python scripts/generate_tabdiff_fraud.py

# Generate adversarial samples (requires trained model)
python scripts/generate_adversarial_fraud.py

# Retrain with existing synthetic/adversarial data
python scripts/retrain_with_synthetic.py
```

## 📈 Output Files

### Adversarial Samples (`outputs/adversarial/`)
- `adversarial_samples.csv` - Raw adversarial examples
- `adversarial_samples_purified.csv` - After TabDiff purification
- `adversarial_samples_purified_clean.csv` - Cleaned for training

### Model Predictions (`outputs/submissions/`)
- `new_submission_model1_*.csv` - Original data only
- `new_submission_model2_*.csv` - Original + Synthetic
- `new_submission_model3_*.csv` - Original + Synthetic + Adversarial

### Visualizations (`outputs/plots/`)
- `comparison_roc_auc.png` - ROC-AUC comparison across models
- `comparison_metrics.png` - Accuracy/Precision/Recall comparison
- `comparison_time.png` - Training time comparison
- `distribution_comparison_multi.png` - Feature distribution plots

## 🔍 Key Features

### Preprocessing Pipeline
- **Data Merging**: Joins transaction and identity datasets
- **Feature Engineering**: Extracts email domains, OS, browser, device info
- **Outlier Removal**: Removes transactions >3σ in amount
- **Encoding**: Label encoding for categoricals, StandardScaler/MinMaxScaler for numerics
- **Dimensionality Reduction**: PCA on V-columns (339 → 9 components)

### TabDiff Integration
- Trains diffusion model on processed fraud samples
- Generates realistic synthetic fraud in PCA space
- Uses EDM (Elucidating Diffusion Models) sampling

### Adversarial Generation
- Iterative noise addition with TabDiff schedule
- Purification via reverse diffusion
- Maintains fraud characteristics while fooling detector

### Model Training
- XGBoost with class balancing (scale_pos_weight)
- Early stopping with validation monitoring
- Sample weighting (3× for adversarial samples)
- Overfitting detection with training curves

## 🧪 Dependencies

See `requirements.txt` for full list. Key packages:
- `xgboost` - Gradient boosting framework
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Preprocessing and metrics
- `torch` - Deep learning (for TabDiff)
- `matplotlib`, `seaborn` - Visualization

## 📝 Notes

- **TabDiff Submodule**: The `TabDiff/` directory is a git submodule. Initialize with:
  ```bash
  git submodule update --init --recursive
  ```

- **Memory Requirements**: PCA transformation requires ~2GB RAM for full dataset

- **GPU Support**: TabDiff training benefits from GPU but works on CPU

## 🤝 Contributing

This is a research project for fraud detection with synthetic data. To modify:

1. Extract shared code to `src/` modules
2. Update scripts to import from `src/`
3. Add new scripts to `scripts/`
4. Update configuration in `src/config.py`

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- IEEE Fraud Detection dataset from Kaggle
- TabDiff: Tabular Data Generation with Diffusion Models