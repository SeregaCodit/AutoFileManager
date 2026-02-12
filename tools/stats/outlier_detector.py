from typing import List

import numpy as np
import pandas as pd

from const_utils.stats_constansts import ImageStatsKeys


class OutlierDetector:
    """
    Service for statistical anomaly detection in datasets.

    This class provides methods to identify outliers using the 3-sigma rule
    (Standard Deviation). It can process features globally or per-class
    to ensure high accuracy in diverse datasets.
    """
    @staticmethod
    def mark_outliers(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Identifies and marks outliers for specified columns in a DataFrame.

        The method calculates the mean and standard deviation for each column.
        Values outside the range [mean - 3*std, mean + 3*std] are marked as 1.
        Columns starting with 'object_' are analyzed within each class group.
        Other columns are analyzed globally.

        Args:
            df (pd.DataFrame): The input feature matrix.
            columns (List[str]): List of numeric column names to analyze.

        Returns:
            pd.DataFrame: The original DataFrame with additional binary
                columns: 'outlier_<col_name>' and 'outlier_any'.
        """
        outlier_marker: str = "outlier_"
        if df.empty:
            return df

        df = df.copy()
        
        for col in columns:

            if col not in df.columns:
                continue

            outlier_column_name = f"{outlier_marker}{col}"

            if col.startswith("object_"):
                class_group = df.groupby(ImageStatsKeys.class_name)[col]
                mean = class_group.transform("mean")
                std = class_group.transform("std")
                upper_limit = mean + 3 * std.values
                lower_limit = mean - 3 * std.values
            else:
                mean = df[col].mean()
                std = df[col].std()
                upper_limit = mean + 3 * std
                lower_limit = mean - 3 * std


            outlier_mask = (df[col].values > upper_limit) | (df[col].values < lower_limit)
            df[outlier_column_name] = outlier_mask.astype("int8")

        outliers = [col for col in df.columns if col.startswith(outlier_marker)]
        df["outlier_any"] = df[outliers].any(axis=1).astype("int8")
        return df