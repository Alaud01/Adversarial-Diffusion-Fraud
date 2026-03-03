"""
Model training and evaluation classes.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    cohen_kappa_score, confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


class applyModel:
    """XGBoost model training and evaluation pipeline."""

    def __init__(self, eval: bool = True):
        self.eval = eval

    def evaluateModel(self, y_test, y_pred, y_pred_proba):
        """Comprehensive model evaluation with multiple metrics."""
        plt.style.use("seaborn-v0_8-whitegrid")

        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5, cbar_kws={"label": "Count"})
        plt.title("Confusion Matrix", fontsize=16)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        print("\n" + "=" * 50)
        print("MODEL EVALUATION METRICS")
        print("=" * 50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"Balanced:  {balanced_acc:.4f}")
        print(f"MCC:       {mcc:.4f}")
        print(f"Kappa:     {kappa:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return [accuracy, precision, recall, f1, roc_auc, balanced_acc, mcc, kappa]

    def plot_training_curves(self, evals_result: dict):
        """Plot training vs validation curves to detect overfitting."""
        plt.style.use("seaborn-v0_8-whitegrid")

        if "validation_0" not in evals_result:
            return

        epochs = len(evals_result["validation_0"]["auc"])
        x_axis = range(epochs)
        train_auc = evals_result["validation_0"]["auc"]
        val_auc = evals_result["validation_1"]["auc"]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x_axis, train_auc, label="Train AUC", color="blue", linewidth=2)
        ax.plot(x_axis, val_auc, label="Validation AUC", color="red", linewidth=2)

        best_iter = np.argmax(val_auc)
        best_val_auc = val_auc[best_iter]
        ax.axvline(x=best_iter, color="green", linestyle="--", label=f"Best ({best_iter})")
        ax.plot(best_iter, best_val_auc, "go", markersize=10)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("AUC Score")
        ax.set_title("Training vs Validation AUC")
        ax.legend()
        ax.grid(True, alpha=0.3)

        gap = train_auc[-1] - val_auc[-1]
        gap_pct = (gap / train_auc[-1]) * 100 if train_auc[-1] > 0 else 0

        textstr = f"Final Gap: {gap:.4f} ({gap_pct:.2f}%)\nBest Val: {best_val_auc:.4f} @ {best_iter}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=props)

        plt.tight_layout()
        plt.show()

        print(f"\n{'='*60}")
        print("OVERFITTING ANALYSIS")
        print(f"{'='*60}")
        print(f"Gap: {gap:.4f} ({gap_pct:.2f}%)")
        if gap_pct > 5:
            print("Significant overfitting detected (>5%)")
        elif gap_pct > 2:
            print("Moderate gap (>2%)")
        else:
            print("Gap acceptable (<2%)")
        print(f"{'='*60}\n")

    def trainModel(
        self,
        X_train_split: pd.DataFrame,
        X_val_split: pd.DataFrame,
        y_train_split: pd.Series,
        y_val_split: pd.Series,
        y_train: pd.Series,
        sample_weight: np.ndarray = None,
        xgb_params: dict = None,
    ):
        """Train XGBoost model with class balancing and early stopping."""
        if xgb_params is None:
            xgb_params = {
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

        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        xgb_params["scale_pos_weight"] = scale_pos_weight

        print(f"  scale_pos_weight: {scale_pos_weight:.4f}")
        print(f"  Training samples - Fraud: {(y_train_split == 1).sum()}, Safe: {(y_train_split == 0).sum()}")

        model = xgb.XGBClassifier(**xgb_params)

        fit_params = {
            "eval_set": [(X_train_split, y_train_split), (X_val_split, y_val_split)],
            "verbose": False,
        }

        if sample_weight is not None:
            weights = self._clean_sample_weights(sample_weight)
            fit_params["sample_weight"] = weights

        model.fit(X_train_split, y_train_split, **fit_params)

        if self.eval:
            self.plot_training_curves(model.evals_result())

        yval_pred = model.predict(X_val_split)
        yval_pred_proba = model.predict_proba(X_val_split)[:, 1]

        if self.eval:
            self.evaluateModel(y_val_split, yval_pred, yval_pred_proba)

        return model

    def _clean_sample_weights(self, sample_weight):
        """Clean sample weights by handling NaN, inf, and non-positive values."""
        if isinstance(sample_weight, pd.Series):
            weights = sample_weight.values.copy()
        else:
            weights = np.array(sample_weight).copy()

        num_invalid = 0
        if np.isnan(weights).any():
            num_invalid += np.isnan(weights).sum()
            weights[np.isnan(weights)] = 1.0
        if np.isinf(weights).any():
            num_invalid += np.isinf(weights).sum()
            weights[np.isinf(weights)] = 1.0
        if (weights <= 0).any():
            num_invalid += (weights <= 0).sum()
            weights[weights <= 0] = 1.0

        if num_invalid > 0:
            print(f"  Fixed {num_invalid} invalid weight values")

        return weights

    def pred_and_submit(
        self,
        model: xgb.XGBClassifier,
        Xtest: pd.DataFrame,
        test_transaction_df: pd.DataFrame,
        plot_pred: bool = True,
        output_path: str = "submission.csv",
    ):
        """Generate predictions and save submission file."""
        y_pred_proba = model.predict_proba(Xtest)[:, 1]

        submission = pd.DataFrame({
            "TransactionID": test_transaction_df["TransactionID"],
            "isFraud": y_pred_proba,
        })

        if plot_pred:
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(10, 6))
            sns.histplot(submission["isFraud"], bins=100, kde=True, color="green", edgecolor="black")
            plt.title("Fraud Probability Predictions")
            plt.xlabel("Predicted Probability")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

        submission.to_csv(output_path, index=False)
        print(f"Saved submission to {output_path}")

        return submission
