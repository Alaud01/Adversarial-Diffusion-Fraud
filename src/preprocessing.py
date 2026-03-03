"""
Data preprocessing classes and utilities.
Extracted from detect_fraud.ipynb for reuse across scripts.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")


class preprocessDatasets:
    """
    Comprehensive preprocessing pipeline for fraud detection data.
    Replicates the exact pipeline from detect_fraud.ipynb.
    """

    def __init__(self):
        self.cat_columns = []
        self.scalers = {}
        self.label_encoders = {}
        self.minmax_scalers = {}

    def join_ID(self, transaction_df: pd.DataFrame, ID_df: pd.DataFrame):
        """Join transaction and identity datasets on TransactionID."""
        merged_df = pd.merge(transaction_df, ID_df, on="TransactionID", how="outer")
        col_list = []
        for col in merged_df.columns:
            if "-" in col:
                col = col.replace("-", "_")
            col_list.append(col)
        merged_df.columns = col_list
        return merged_df

    def replace_blanks(self, df: pd.DataFrame):
        """Replace missing values with placeholder -999."""
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(-999)
            else:
                df[col] = df[col].fillna("-999")
        return df

    def encode_df(self, df: pd.DataFrame):
        """Encode and scale dataframe columns appropriately."""
        for col in df.columns:
            if col in ["TransactionID", "isFraud"]:
                continue

            elif re.match(r"^D\d+$", col):
                if col not in self.minmax_scalers:
                    self.minmax_scalers[col] = MinMaxScaler()
                    df[col] = self.minmax_scalers[col].fit_transform(df[[col]])
                else:
                    df[col] = self.minmax_scalers[col].transform(df[[col]])

            elif pd.api.types.is_numeric_dtype(df[col]):
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df[col] = self.scalers[col].fit_transform(df[[col]])
                else:
                    df[col] = self.scalers[col].transform(df[[col]])

            else:
                if col not in self.cat_columns:
                    self.cat_columns.append(col)

                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    le = self.label_encoders[col]
                    valid_classes = set(le.classes_)
                    values = df[col].astype(str).values
                    fallback_val = le.classes_[0]
                    safe_values = [x if x in valid_classes else fallback_val for x in values]
                    df[col] = le.transform(safe_values)

        return df

    def remove_outliers(self, data: pd.DataFrame, column: str, threshold: float = 3.0):
        """Remove outliers based on z-score threshold."""
        z_scores = stats.zscore(data[column])
        non_outliers = np.abs(z_scores) < threshold
        original_len = len(data)
        data = data[non_outliers]
        final_len = len(data)
        print(f"{original_len - final_len} outliers removed (z-score > {threshold})")
        return data

    def remove_empty_cols(self, data: pd.DataFrame, threshold: float = 0.90):
        """Drop columns with more than threshold% missing values."""
        data.dropna(thresh=int((1 - threshold) * len(data)), axis=1, inplace=True)
        return data

    def feature_engineer(self, data: pd.DataFrame):
        """Extract additional features from existing columns."""
        # Email domain processing
        if "P_emaildomain" in data.columns:
            email_p_str = data["P_emaildomain"].astype(str)
            split_p = email_p_str.str.split(".", n=1, expand=True)
            if split_p.shape[1] == 1:
                split_p[1] = ""
            data["P_emailserver"] = split_p[0].fillna("")
            data["P_suffix"] = split_p[1].fillna("")

        if "R_emaildomain" in data.columns:
            email_r_str = data["R_emaildomain"].astype(str)
            split_r = email_r_str.str.split(".", n=1, expand=True)
            if split_r.shape[1] == 1:
                split_r[1] = ""
            data["R_emailserver"] = split_r[0].fillna("")
            data["R_suffix"] = split_r[1].fillna("")

        # OS extraction
        if "id_30" in data.columns:
            id_30_str = data["id_30"].astype(str)
            data["os"] = id_30_str.str.split(" ", expand=True)[0].fillna("")

        # Screen dimensions
        if "id_33" in data.columns:
            id_33_str = data["id_33"].astype(str)
            split_screen = id_33_str.str.split("x", expand=True)
            if split_screen.shape[1] >= 1:
                data["screen_width"] = pd.to_numeric(split_screen[0], errors="coerce")
            else:
                data["screen_width"] = None
            if split_screen.shape[1] >= 2:
                data["screen_height"] = pd.to_numeric(split_screen[1], errors="coerce")
            else:
                data["screen_height"] = None

        # Browser and device
        if "id_31" in data.columns:
            id_31_str = data["id_31"].astype(str)
            data["browser"] = id_31_str.str.split(" ", expand=True)[0].str.lower().fillna("")

        if "DeviceInfo" in data.columns:
            device_str = data["DeviceInfo"].astype(str)
            data["device_name"] = device_str.str.split(" ", expand=True)[0].str.lower().fillna("")

        # Standardize browser and device names
        data = self._standardize_browser_device(data)

        # Temporal features
        if "TransactionDT" in data.columns:
            start_date = datetime(2017, 11, 30)
            data["TransactionFullDate"] = data["TransactionDT"].apply(
                lambda x: start_date + timedelta(seconds=x)
            )
            data["TransactionDate"] = data["TransactionFullDate"].dt.date
            data["DayOfWeek"] = data["TransactionFullDate"].dt.dayofweek.apply(lambda x: (x + 1) % 7)
            data["HourOfDay"] = data["TransactionFullDate"].dt.hour
            data["Month"] = data["TransactionFullDate"].dt.month

        return data

    def _standardize_browser_device(self, data: pd.DataFrame):
        """Standardize browser and device names using regex patterns."""
        def match_patterns(df: pd.DataFrame, patterns: dict, col_name: str):
            for pattern, value in patterns.items():
                df[col_name] = df[col_name].str.replace(pattern, value, regex=True)
            return df

        browser_patterns = {
            r"samsung/sm-g532m|samsung/sch|samsung/sm-g531h": "samsung",
            r"generic/android": "android",
            r"mozilla/firefox": "firefox",
            r"nokia/lumia": "nokia",
            r"zte/blade": "zte",
            r"lg/k-200": "lg",
            r"lanix/ilium": "lanix",
            r"blu/dash": "blu",
            r"m4tel/m4": "m4",
        }

        device_patterns = {
            r"samsung|sgh|sm|gt-": "samsung",
            r"mot": "motorola",
            r"ale-|.*-l|hi": "huawei",
            r"lg": "lg",
            r"rv:": "rv",
            r"blade": "zte",
            r"xt": "sony",
            r"iphone": "ios",
            r"lenovo": "lenovo",
            r"mi|redmi": "xiaomi",
            r"ilium": "ilium",
            r"alcatel": "alcatel",
            r"asus": "asus",
        }

        if "browser" in data.columns:
            data = match_patterns(data, browser_patterns, "browser")
        if "device_name" in data.columns:
            data = match_patterns(data, device_patterns, "device_name")

        return data

    def reduce_memory(self, df: pd.DataFrame):
        """Optimize memory usage by downcasting numeric types."""
        start = df.memory_usage().sum() / 1024**2
        print(f"Starting memory usage: {start:.2f} MB")

        for col in df.columns:
            col_type = df[col].dtype

            if str(col_type).startswith("float"):
                col_min = df[col].min()
                col_max = df[col].max()
                if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

            elif str(col_type).startswith("int"):
                col_min = df[col].min()
                col_max = df[col].max()
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype("category")

        end = df.memory_usage().sum() / 1024**2
        print(f"Memory after downsizing: {end:.2f} MB")
        print(f"Memory reduction: {100 * (start - end) / start:.1f}%")
        return df

    def final_preprocessing(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Final preprocessing: sort, drop unnecessary columns, align columns."""
        if "TransactionID" in train_df.columns:
            train_df = train_df.sort_values(by="TransactionID", ascending=True).reset_index(drop=True)
        if "TransactionID" in test_df.columns:
            test_df = test_df.sort_values(by="TransactionID", ascending=True).reset_index(drop=True)

        cols_to_drop = [
            "P_emaildomain", "R_emaildomain", "id_30", "id_31", "id_33",
            "DeviceInfo", "TransactionDT", "TransactionFullDate", "TransactionDate", "TransactionID",
        ]

        train_cols_to_drop = [c for c in cols_to_drop if c in train_df.columns]
        test_cols_to_drop = [c for c in cols_to_drop if c in test_df.columns]

        train_df = train_df.drop(columns=train_cols_to_drop)
        test_df = test_df.drop(columns=test_cols_to_drop)

        col_drop = [col for col in test_df.columns if col not in train_df.columns]
        dropped_df = test_df.drop(columns=col_drop)

        return train_df, dropped_df


class ReduceDimension:
    """PCA-based dimensionality reduction for V-columns."""

    def __init__(self, traindata: pd.DataFrame, testdata: pd.DataFrame):
        self.traindata = traindata
        self.testdata = testdata
        self.v_cols = []
        self.pca = None
        self.comp_90 = None

    def plot_and_reduceD(self, plots: bool = False, variance_threshold: float = 0.90):
        """Apply PCA to V-columns and reduce dimensionality."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.style.use("seaborn-v0_8-whitegrid")

        self.v_cols = [col for col in self.traindata.columns if re.match(r"^V\d+$", col)]

        if not self.v_cols:
            print("Warning: No V-columns found for PCA. Skipping.")
            return self.traindata, self.testdata

        v_data = self.traindata[self.v_cols]
        self.pca = PCA().fit(v_data)
        v_data_pca = self.pca.transform(v_data)
        explained_variance = self.pca.explained_variance_ratio_.cumsum()

        v_test_data = self.testdata[self.v_cols]
        v_test_data_pca = self.pca.transform(v_test_data)

        self.comp_90 = next(i for i, total in enumerate(explained_variance) if total >= variance_threshold) + 1
        print(f"\nPCA Components for {variance_threshold*100:.0f}% variance: {self.comp_90}")
        print(f"Explained variance: {explained_variance[self.comp_90-1]:.4f}")

        pc_cols = [f"PC{i+1}" for i in range(self.comp_90)]
        v_data_pca_df = pd.DataFrame(v_data_pca[:, : self.comp_90], columns=pc_cols)
        v_test_data_pca_df = pd.DataFrame(v_test_data_pca[:, : self.comp_90], columns=pc_cols)

        data_final = pd.concat(
            [self.traindata.drop(columns=self.v_cols).reset_index(drop=True), v_data_pca_df], axis=1
        )
        test_data_final = pd.concat(
            [self.testdata.drop(columns=self.v_cols).reset_index(drop=True), v_test_data_pca_df], axis=1
        )

        if plots:
            self._plot_pca(explained_variance)

        return data_final, test_data_final

    def _plot_pca(self, explained_variance: np.ndarray):
        """Generate PCA visualization plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 7))
        sns.lineplot(
            x=range(1, len(self.pca.explained_variance_ratio_) + 1),
            y=self.pca.explained_variance_ratio_,
            marker="o", linestyle="--", color="darkblue",
        )
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Scree Plot")
        plt.show()

        plt.figure(figsize=(12, 7))
        sns.lineplot(
            x=range(1, len(explained_variance) + 1),
            y=explained_variance, marker="o", linestyle="-", color="darkgreen",
        )
        plt.axhline(y=0.90, color="red", linestyle="--", label="90% Threshold")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Cumulative Explained Variance")
        plt.legend()
        plt.show()
