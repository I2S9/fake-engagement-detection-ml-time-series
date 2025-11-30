"""
Generate synthetic time series data for fake engagement detection.

This module creates realistic time series of engagement metrics (views, likes,
comments, shares) with both normal and fake patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple
import random


def generate_normal_timeseries(
    start_date: datetime,
    length_days: int,
    frequency: str = "H",
    video_id: str = "normal_001",
    base_views: float = 1000.0,
    base_likes: float = 50.0,
    base_comments: float = 10.0,
    base_shares: float = 5.0,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a normal engagement time series with realistic patterns.

    Parameters
    ----------
    start_date : datetime
        Start date for the time series
    length_days : int
        Number of days to generate
    frequency : str
        Frequency of data points ('H' for hourly, 'D' for daily)
    video_id : str
        Unique identifier for the video
    base_views : float
        Base level of views per time unit
    base_likes : float
        Base level of likes per time unit
    base_comments : float
        Base level of comments per time unit
    base_shares : float
        Base level of shares per time unit
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Time series with columns: id, timestamp, views, likes, comments, shares, label
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    if frequency == "H":
        periods = length_days * 24
        date_range = pd.date_range(start=start_date, periods=periods, freq="h")
    elif frequency == "D":
        periods = length_days
        date_range = pd.date_range(start=start_date, periods=periods, freq="D")
    else:
        raise ValueError("frequency must be 'H' (hourly) or 'D' (daily)")

    n_points = len(date_range)

    # create base trend with natural decay after initial publication
    time_index = np.arange(n_points)
    decay_factor = np.exp(-time_index / (n_points * 0.3))
    trend = 1.0 + 0.5 * decay_factor

    # add day/night cycle for hourly data
    if frequency == "H":
        hour_of_day = np.array([d.hour for d in date_range])
        day_night_cycle = 0.3 + 0.7 * (
            1.0
            + 0.4 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        )
        day_night_cycle = np.clip(day_night_cycle, 0.2, 1.5)
    else:
        day_night_cycle = np.ones(n_points)

    # add weekend effect
    is_weekend = np.array([d.weekday() >= 5 for d in date_range])
    weekend_multiplier = np.where(is_weekend, 1.3, 1.0)

    # add random noise
    noise_views = np.random.normal(1.0, 0.15, n_points)
    noise_likes = np.random.normal(1.0, 0.12, n_points)
    noise_comments = np.random.normal(1.0, 0.20, n_points)
    noise_shares = np.random.normal(1.0, 0.18, n_points)

    # combine all effects
    views_multiplier = trend * day_night_cycle * weekend_multiplier * noise_views
    likes_multiplier = trend * day_night_cycle * weekend_multiplier * noise_likes
    comments_multiplier = trend * day_night_cycle * weekend_multiplier * noise_comments
    shares_multiplier = trend * day_night_cycle * weekend_multiplier * noise_shares

    # generate metrics with realistic ratios
    views = np.maximum(0, base_views * views_multiplier).astype(int)
    likes = np.maximum(0, base_likes * likes_multiplier).astype(int)
    comments = np.maximum(0, base_comments * comments_multiplier).astype(int)
    shares = np.maximum(0, base_shares * shares_multiplier).astype(int)

    # ensure likes <= views, comments <= views, shares <= views
    likes = np.minimum(likes, views)
    comments = np.minimum(comments, views)
    shares = np.minimum(shares, views)

    # add some correlation but not perfect
    likes = (likes * 0.7 + views * 0.05 * np.random.uniform(0.8, 1.2, n_points)).astype(int)
    likes = np.minimum(likes, views)

    df = pd.DataFrame(
        {
            "id": video_id,
            "timestamp": date_range,
            "views": views,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "label": "normal",
        }
    )

    return df


def generate_fake_timeseries(
    start_date: datetime,
    length_days: int,
    frequency: str = "H",
    video_id: str = "fake_001",
    base_views: float = 1000.0,
    base_likes: float = 50.0,
    base_comments: float = 10.0,
    base_shares: float = 5.0,
    fake_pattern: str = "burst",
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a fake engagement time series with suspicious patterns.

    Parameters
    ----------
    start_date : datetime
        Start date for the time series
    length_days : int
        Number of days to generate
    frequency : str
        Frequency of data points ('H' for hourly, 'D' for daily)
    video_id : str
        Unique identifier for the video
    base_views : float
        Base level of views per time unit
    base_likes : float
        Base level of likes per time unit
    base_comments : float
        Base level of comments per time unit
    base_shares : float
        Base level of shares per time unit
    fake_pattern : str
        Type of fake pattern: 'burst', 'synchronized', 'off_peak', 'perfect_correlation'
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Time series with columns: id, timestamp, views, likes, comments, shares, label
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    if frequency == "H":
        periods = length_days * 24
        date_range = pd.date_range(start=start_date, periods=periods, freq="h")
    elif frequency == "D":
        periods = length_days
        date_range = pd.date_range(start=start_date, periods=periods, freq="D")
    else:
        raise ValueError("frequency must be 'H' (hourly) or 'D' (daily)")

    n_points = len(date_range)

    # start with a normal baseline
    baseline_views = base_views * np.ones(n_points)
    baseline_likes = base_likes * np.ones(n_points)
    baseline_comments = base_comments * np.ones(n_points)
    baseline_shares = base_shares * np.ones(n_points)

    if fake_pattern == "burst":
        # sudden impossible spikes
        n_bursts = random.randint(2, 5)
        burst_locations = random.sample(range(n_points), n_bursts)
        for loc in burst_locations:
            burst_size = random.uniform(5.0, 20.0)
            burst_duration = random.randint(1, 3)
            for i in range(burst_duration):
                if loc + i < n_points:
                    baseline_views[loc + i] *= burst_size
                    baseline_likes[loc + i] *= burst_size
                    baseline_comments[loc + i] *= burst_size
                    baseline_shares[loc + i] *= burst_size

    elif fake_pattern == "synchronized":
        # perfect synchronization across all metrics
        sync_multiplier = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(n_points) / 12)
        baseline_views *= sync_multiplier
        baseline_likes *= sync_multiplier
        baseline_comments *= sync_multiplier
        baseline_shares *= sync_multiplier

    elif fake_pattern == "off_peak":
        # regular bursts during off-peak hours (e.g., 2-4 AM)
        if frequency == "H":
            hour_of_day = np.array([d.hour for d in date_range])
            off_peak_mask = (hour_of_day >= 2) & (hour_of_day <= 4)
            baseline_views[off_peak_mask] *= random.uniform(3.0, 8.0)
            baseline_likes[off_peak_mask] *= random.uniform(3.0, 8.0)
            baseline_comments[off_peak_mask] *= random.uniform(3.0, 8.0)
            baseline_shares[off_peak_mask] *= random.uniform(3.0, 8.0)
        else:
            # for daily data, use random days
            n_off_peak = random.randint(3, 7)
            off_peak_days = random.sample(range(n_points), n_off_peak)
            baseline_views[off_peak_days] *= random.uniform(3.0, 8.0)
            baseline_likes[off_peak_days] *= random.uniform(3.0, 8.0)
            baseline_comments[off_peak_days] *= random.uniform(3.0, 8.0)
            baseline_shares[off_peak_days] *= random.uniform(3.0, 8.0)

    elif fake_pattern == "perfect_correlation":
        # perfect correlation between metrics (unrealistic)
        correlation_signal = 1.0 + 0.8 * np.sin(2 * np.pi * np.arange(n_points) / 24)
        baseline_views *= correlation_signal
        baseline_likes *= correlation_signal
        baseline_comments *= correlation_signal
        baseline_shares *= correlation_signal

    else:
        raise ValueError(
            f"Unknown fake_pattern: {fake_pattern}. "
            "Must be one of: burst, synchronized, off_peak, perfect_correlation"
        )

    # add minimal noise to make it slightly more realistic
    noise_views = np.random.normal(1.0, 0.05, n_points)
    noise_likes = np.random.normal(1.0, 0.05, n_points)
    noise_comments = np.random.normal(1.0, 0.05, n_points)
    noise_shares = np.random.normal(1.0, 0.05, n_points)

    views = np.maximum(0, baseline_views * noise_views).astype(int)
    likes = np.maximum(0, baseline_likes * noise_likes).astype(int)
    comments = np.maximum(0, baseline_comments * noise_comments).astype(int)
    shares = np.maximum(0, baseline_shares * noise_shares).astype(int)

    # ensure logical constraints
    likes = np.minimum(likes, views)
    comments = np.minimum(comments, views)
    shares = np.minimum(shares, views)

    df = pd.DataFrame(
        {
            "id": video_id,
            "timestamp": date_range,
            "views": views,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "label": "fake",
        }
    )

    return df


def generate_dataset(
    n_normal: int = 100,
    n_fake: int = 30,
    length_days: int = 30,
    frequency: str = "H",
    start_date: Optional[datetime] = None,
    output_path: str = "data/raw/engagement_timeseries.parquet",
    output_format: str = "parquet",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a complete dataset with normal and fake time series.

    Parameters
    ----------
    n_normal : int
        Number of normal time series to generate
    n_fake : int
        Number of fake time series to generate
    length_days : int
        Length of each time series in days
    frequency : str
        Frequency of data points ('H' for hourly, 'D' for daily)
    start_date : datetime, optional
        Start date for all time series. If None, uses current date.
    output_path : str
        Path to save the generated dataset
    output_format : str
        Format to save: 'parquet' or 'csv'
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Combined dataset with all time series
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    if start_date is None:
        start_date = datetime.now() - timedelta(days=length_days)

    fake_patterns = ["burst", "synchronized", "off_peak", "perfect_correlation"]

    all_dataframes = []

    # generate normal time series
    for i in range(n_normal):
        video_id = f"normal_{i+1:03d}"
        base_views = np.random.uniform(500, 5000)
        base_likes = np.random.uniform(20, 200)
        base_comments = np.random.uniform(5, 50)
        base_shares = np.random.uniform(2, 25)

        ts_start = start_date + timedelta(
            days=random.randint(0, max(1, length_days - 7))
        )

        df = generate_normal_timeseries(
            start_date=ts_start,
            length_days=length_days,
            frequency=frequency,
            video_id=video_id,
            base_views=base_views,
            base_likes=base_likes,
            base_comments=base_comments,
            base_shares=base_shares,
            random_seed=random_seed + i,
        )
        all_dataframes.append(df)

    # generate fake time series
    for i in range(n_fake):
        video_id = f"fake_{i+1:03d}"
        base_views = np.random.uniform(500, 5000)
        base_likes = np.random.uniform(20, 200)
        base_comments = np.random.uniform(5, 50)
        base_shares = np.random.uniform(2, 25)

        fake_pattern = random.choice(fake_patterns)
        ts_start = start_date + timedelta(
            days=random.randint(0, max(1, length_days - 7))
        )

        df = generate_fake_timeseries(
            start_date=ts_start,
            length_days=length_days,
            frequency=frequency,
            video_id=video_id,
            base_views=base_views,
            base_likes=base_likes,
            base_comments=base_comments,
            base_shares=base_shares,
            fake_pattern=fake_pattern,
            random_seed=random_seed + n_normal + i,
        )
        all_dataframes.append(df)

    # combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # save to file
    if output_format == "parquet":
        combined_df.to_parquet(output_path, index=False)
    elif output_format == "csv":
        combined_df.to_csv(output_path, index=False)
    else:
        raise ValueError("output_format must be 'parquet' or 'csv'")

    print(f"Generated dataset with {n_normal} normal and {n_fake} fake time series")
    print(f"Total rows: {len(combined_df)}")
    print(f"Saved to: {output_path}")

    return combined_df


if __name__ == "__main__":
    # example usage
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic engagement time series")
    parser.add_argument(
        "--n_normal", type=int, default=100, help="Number of normal time series"
    )
    parser.add_argument(
        "--n_fake", type=int, default=30, help="Number of fake time series"
    )
    parser.add_argument(
        "--length_days", type=int, default=30, help="Length of each series in days"
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="H",
        choices=["H", "D"],
        help="Frequency: H (hourly) or D (daily)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/raw/engagement_timeseries.parquet",
        help="Output file path",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="Output format: parquet or csv",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    generate_dataset(
        n_normal=args.n_normal,
        n_fake=args.n_fake,
        length_days=args.length_days,
        frequency=args.frequency,
        output_path=args.output_path,
        output_format=args.output_format,
        random_seed=args.random_seed,
    )

