"""
Temporal feature engineering for engagement time series.

This module provides functions to extract temporal features from time series data,
including rolling statistics, ratios, burst detection, autocorrelation, and entropy.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Union
from scipy import stats
from scipy.signal import find_peaks


def compute_rolling_statistics(
    series: pd.Series,
    window_sizes: List[int] = [6, 12, 24],
    metrics: List[str] = ["mean", "std", "min", "max"],
) -> pd.DataFrame:
    """
    Compute rolling statistics for a time series.

    Parameters
    ----------
    series : pd.Series
        Input time series
    window_sizes : list of int
        Window sizes for rolling calculations (in time units)
    metrics : list of str
        Statistics to compute: 'mean', 'std', 'min', 'max', 'median'

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling statistics as columns
    """
    features = {}

    for window in window_sizes:
        if len(series) < window:
            # if series is shorter than window, use available data
            window = max(1, len(series) - 1)

        rolling = series.rolling(window=window, min_periods=1)

        for metric in metrics:
            if metric == "mean":
                features[f"rolling_mean_{window}"] = rolling.mean()
            elif metric == "std":
                features[f"rolling_std_{window}"] = rolling.std()
            elif metric == "min":
                features[f"rolling_min_{window}"] = rolling.min()
            elif metric == "max":
                features[f"rolling_max_{window}"] = rolling.max()
            elif metric == "median":
                features[f"rolling_median_{window}"] = rolling.median()

    return pd.DataFrame(features, index=series.index)


def compute_ratios(
    df: pd.DataFrame,
    numerator_cols: List[str],
    denominator_col: str = "views",
) -> pd.DataFrame:
    """
    Compute ratio features (e.g., likes/views, comments/views).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with engagement metrics
    numerator_cols : list of str
        Columns to use as numerators
    denominator_col : str
        Column to use as denominator

    Returns
    -------
    pd.DataFrame
        DataFrame with ratio features
    """
    features = {}

    for num_col in numerator_cols:
        if num_col in df.columns and denominator_col in df.columns:
            ratio = df[num_col] / (df[denominator_col] + 1e-6)
            features[f"ratio_{num_col}_{denominator_col}"] = ratio

    return pd.DataFrame(features, index=df.index)


def detect_bursts(
    series: pd.Series,
    threshold_multiplier: float = 2.0,
    min_peak_distance: int = 3,
) -> Dict[str, Union[int, float]]:
    """
    Detect bursts (peaks) in a time series.

    Parameters
    ----------
    series : pd.Series
        Input time series
    threshold_multiplier : float
        Multiplier for mean to determine peak threshold
    min_peak_distance : int
        Minimum distance between peaks

    Returns
    -------
    dict
        Dictionary with burst-related features
    """
    if len(series) < 2:
        return {
            "n_peaks": 0,
            "max_value": float(series.iloc[0]) if len(series) > 0 else 0.0,
            "mean_value": float(series.mean()) if len(series) > 0 else 0.0,
            "max_mean_ratio": 0.0,
            "peak_intensity_mean": 0.0,
        }

    mean_val = series.mean()
    std_val = series.std()
    max_val = series.max()

    # find peaks using scipy
    threshold = mean_val + threshold_multiplier * std_val
    peaks, properties = find_peaks(
        series.values,
        height=threshold,
        distance=min_peak_distance,
    )

    n_peaks = len(peaks)
    max_mean_ratio = max_val / (mean_val + 1e-6)

    # compute average peak intensity
    if n_peaks > 0:
        peak_values = series.iloc[peaks].values
        peak_intensity_mean = np.mean(peak_values) / (mean_val + 1e-6)
    else:
        peak_intensity_mean = 0.0

    return {
        "n_peaks": n_peaks,
        "max_value": float(max_val),
        "mean_value": float(mean_val),
        "max_mean_ratio": float(max_mean_ratio),
        "peak_intensity_mean": float(peak_intensity_mean),
    }


def compute_autocorrelation(
    series: pd.Series,
    lags: List[int] = [1, 6, 12, 24],
) -> Dict[str, float]:
    """
    Compute autocorrelation at different lags.

    Parameters
    ----------
    series : pd.Series
        Input time series
    lags : list of int
        Lags to compute autocorrelation for

    Returns
    -------
    dict
        Dictionary with autocorrelation features
    """
    features = {}

    if len(series) < 2:
        for lag in lags:
            features[f"autocorr_lag_{lag}"] = 0.0
        return features

    # remove NaN values
    series_clean = series.dropna()

    if len(series_clean) < 2:
        for lag in lags:
            features[f"autocorr_lag_{lag}"] = 0.0
        return features

    for lag in lags:
        if lag >= len(series_clean):
            features[f"autocorr_lag_{lag}"] = 0.0
            continue

        # compute autocorrelation
        autocorr = series_clean.autocorr(lag=lag)
        if pd.isna(autocorr):
            autocorr = 0.0

        features[f"autocorr_lag_{lag}"] = float(autocorr)

    return features


def compute_entropy(
    series: pd.Series,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute entropy and regularity measures.

    Parameters
    ----------
    series : pd.Series
        Input time series
    n_bins : int
        Number of bins for histogram-based entropy

    Returns
    -------
    dict
        Dictionary with entropy features
    """
    features = {}

    if len(series) < 2:
        return {
            "entropy": 0.0,
            "regularity": 0.0,
            "coefficient_of_variation": 0.0,
        }

    series_clean = series.dropna()

    if len(series_clean) < 2:
        return {
            "entropy": 0.0,
            "regularity": 0.0,
            "coefficient_of_variation": 0.0,
        }

    # shannon entropy from histogram
    hist, _ = np.histogram(series_clean, bins=n_bins)
    hist = hist + 1e-10
    prob = hist / hist.sum()
    entropy = -np.sum(prob * np.log2(prob))
    features["entropy"] = float(entropy)

    # regularity (inverse of coefficient of variation)
    mean_val = series_clean.mean()
    std_val = series_clean.std()
    if mean_val > 0:
        cv = std_val / mean_val
        regularity = 1.0 / (1.0 + cv)
        features["coefficient_of_variation"] = float(cv)
    else:
        regularity = 0.0
        features["coefficient_of_variation"] = 0.0

    features["regularity"] = float(regularity)

    return features


def compute_trend_features(
    series: pd.Series,
) -> Dict[str, float]:
    """
    Compute trend-related features.

    Parameters
    ----------
    series : pd.Series
        Input time series

    Returns
    -------
    dict
        Dictionary with trend features
    """
    features = {}

    if len(series) < 2:
        return {
            "trend_slope": 0.0,
            "trend_strength": 0.0,
            "first_last_ratio": 0.0,
        }

    series_clean = series.dropna()

    if len(series_clean) < 2:
        return {
            "trend_slope": 0.0,
            "trend_strength": 0.0,
            "first_last_ratio": 0.0,
        }

    # linear trend
    x = np.arange(len(series_clean))
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x, series_clean.values
    )

    features["trend_slope"] = float(slope)
    features["trend_strength"] = float(r_value**2)

    # first to last ratio
    first_val = series_clean.iloc[0]
    last_val = series_clean.iloc[-1]
    if first_val > 0:
        features["first_last_ratio"] = float(last_val / first_val)
    else:
        features["first_last_ratio"] = 0.0

    return features


def extract_temporal_features(
    df: pd.DataFrame,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    metric_columns: Optional[List[str]] = None,
    window_sizes: List[int] = [6, 12, 24],
    autocorr_lags: List[int] = [1, 6, 12, 24],
    aggregate_per_id: bool = True,
) -> pd.DataFrame:
    """
    Extract all temporal features for a time series dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data
    id_column : str
        Name of the ID column
    timestamp_column : str
        Name of the timestamp column
    metric_columns : list of str, optional
        Columns to extract features from. If None, uses: views, likes, comments, shares
    window_sizes : list of int
        Window sizes for rolling statistics
    autocorr_lags : list of int
        Lags for autocorrelation
    aggregate_per_id : bool
        If True, aggregate features per ID. If False, return features per time point.

    Returns
    -------
    pd.DataFrame
        DataFrame with extracted features
    """
    if metric_columns is None:
        metric_columns = ["views", "likes", "comments", "shares"]
        metric_columns = [col for col in metric_columns if col in df.columns]

    all_features = []

    for video_id, group in df.groupby(id_column):
        group = group.sort_values(timestamp_column).reset_index(drop=True)

        video_features = {}

        # extract features for each metric
        for metric in metric_columns:
            if metric not in group.columns:
                continue

            series = group[metric]

            # rolling statistics
            rolling_features = compute_rolling_statistics(
                series, window_sizes=window_sizes
            )

            # burst features
            burst_features = detect_bursts(series)

            # autocorrelation
            autocorr_features = compute_autocorrelation(series, lags=autocorr_lags)

            # entropy
            entropy_features = compute_entropy(series)

            # trend features
            trend_features = compute_trend_features(series)

            # combine all features with metric prefix
            for col in rolling_features.columns:
                video_features[f"{metric}_{col}"] = rolling_features[col].values

            for key, value in burst_features.items():
                video_features[f"{metric}_{key}"] = value

            for key, value in autocorr_features.items():
                video_features[f"{metric}_{key}"] = value

            for key, value in entropy_features.items():
                video_features[f"{metric}_{key}"] = value

            for key, value in trend_features.items():
                video_features[f"{metric}_{key}"] = value

        # ratio features
        ratio_features = compute_ratios(
            group, numerator_cols=["likes", "comments", "shares"], denominator_col="views"
        )

        for col in ratio_features.columns:
            video_features[col] = ratio_features[col].values

        # aggregate per ID if requested
        if aggregate_per_id:
            # aggregate rolling features (take mean of rolling stats)
            aggregated = {}
            for key, value in video_features.items():
                if isinstance(value, np.ndarray):
                    if "rolling" in key:
                        aggregated[key] = np.nanmean(value)
                    elif "ratio" in key:
                        aggregated[key] = np.nanmean(value)
                    else:
                        aggregated[key] = value[0] if len(value) > 0 else 0.0
                else:
                    aggregated[key] = value

            aggregated[id_column] = video_id

            # add label if available
            if "label" in group.columns:
                aggregated["label"] = group["label"].iloc[0]

            all_features.append(aggregated)
        else:
            # return features per time point
            feature_df = pd.DataFrame(video_features, index=group.index)
            feature_df[id_column] = video_id
            feature_df[timestamp_column] = group[timestamp_column]

            if "label" in group.columns:
                feature_df["label"] = group["label"]

            all_features.append(feature_df)

    if aggregate_per_id:
        result_df = pd.DataFrame(all_features)
    else:
        result_df = pd.concat(all_features, ignore_index=True)

    return result_df


def save_features(
    features_df: pd.DataFrame,
    output_path: str,
    output_format: str = "parquet",
) -> None:
    """
    Save extracted features to file.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with features
    output_path : str
        Path to save the features
    output_format : str
        Format to save: 'parquet' or 'csv'
    """
    if output_format == "parquet":
        features_df.to_parquet(output_path, index=False)
    elif output_format == "csv":
        features_df.to_csv(output_path, index=False)
    else:
        raise ValueError("output_format must be 'parquet' or 'csv'")

    print(f"Features saved to {output_path}")
    print(f"Shape: {features_df.shape}")

