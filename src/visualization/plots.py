# src/visualization/plots.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def plot_series_with_anomalies(timestamps, values, anomaly_mask, title=None, 
                                show_zones=True, spike_color='red', normal_color='blue'):
    """
    Plot time series with anomalies highlighted in red.
    
    Args:
        timestamps: Time values
        values: Series values
        anomaly_mask: Boolean mask for anomalies
        title: Plot title
        show_zones: If True, highlight anomaly zones with red background
        spike_color: Color for anomaly spikes (default: red)
        normal_color: Color for normal series (default: blue)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot normal series in blue
    ax.plot(timestamps, values, label="Normal Series", linewidth=2, 
            color=normal_color, alpha=0.7, zorder=1)

    if anomaly_mask is not None and anomaly_mask.any():
        # Highlight anomaly zones with red background
        if show_zones:
            anomaly_indices = np.where(anomaly_mask)[0]
            if len(anomaly_indices) > 0:
                # Group consecutive anomalies
                zones = []
                start = anomaly_indices[0]
                for i in range(1, len(anomaly_indices)):
                    if anomaly_indices[i] != anomaly_indices[i-1] + 1:
                        zones.append((start, anomaly_indices[i-1]))
                        start = anomaly_indices[i]
                zones.append((start, anomaly_indices[-1]))
                
                # Draw red zones
                for start_idx, end_idx in zones:
                    if start_idx < len(timestamps) and end_idx < len(timestamps):
                        ax.axvspan(timestamps[start_idx], timestamps[end_idx], 
                                  alpha=0.3, color=spike_color, zorder=0, 
                                  label="Anomaly Zone" if start_idx == zones[0][0] else "")
        
        # Plot anomaly spikes in red
        ax.scatter(
            np.array(timestamps)[anomaly_mask],
            np.array(values)[anomaly_mask],
            marker="o",
            s=100,
            color=spike_color,
            label="Anomaly Spike",
            zorder=5,
            edgecolors='darkred',
            linewidths=2
        )

    ax.set_xlabel("Time", fontsize=12, fontweight='bold')
    ax.set_ylabel("Value", fontsize=12, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_score_with_threshold(timestamps, scores, threshold, title=None, 
                              show_zones=True, alert_color='red'):
    """
    Plot anomaly scores with threshold and highlight alert zones in red.
    
    Args:
        timestamps: Time values
        scores: Anomaly scores
        threshold: Alert threshold
        title: Plot title
        show_zones: If True, highlight zones above threshold with red background
        alert_color: Color for alert zones (default: red)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot scores
    ax.plot(timestamps, scores, label="Anomaly Score", linewidth=2, 
            color='orange', alpha=0.8, zorder=2)
    
    # Highlight alert zones above threshold
    if show_zones:
        alert_mask = np.array(scores) > threshold
        if alert_mask.any():
            alert_indices = np.where(alert_mask)[0]
            if len(alert_indices) > 0:
                # Group consecutive alerts
                zones = []
                start = alert_indices[0]
                for i in range(1, len(alert_indices)):
                    if alert_indices[i] != alert_indices[i-1] + 1:
                        zones.append((start, alert_indices[i-1]))
                        start = alert_indices[i]
                zones.append((start, alert_indices[-1]))
                
                # Draw red alert zones
                for start_idx, end_idx in zones:
                    if start_idx < len(timestamps) and end_idx < len(timestamps):
                        ax.axvspan(timestamps[start_idx], timestamps[end_idx], 
                                  alpha=0.4, color=alert_color, zorder=0,
                                  label="Alert Zone" if start_idx == zones[0][0] else "")
    
    # Threshold line
    ax.axhline(threshold, linestyle="--", linewidth=2, color='red', 
               label=f"Alert Threshold ({threshold:.2f})", zorder=3)
    
    # Fill area above threshold
    ax.fill_between(timestamps, threshold, scores, where=(np.array(scores) > threshold),
                   alpha=0.2, color=alert_color, zorder=1)

    ax.set_xlabel("Time", fontsize=12, fontweight='bold')
    ax.set_ylabel("Anomaly Score", fontsize=12, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(1.0, max(scores) * 1.1)])
    fig.tight_layout()
    return fig, ax


def plot_reconstruction(original, reconstructed, anomaly_mask=None, title=None,
                        error_threshold=None, show_error_zones=True):
    """
    Plot original vs reconstructed series with error zones highlighted in red.
    
    Args:
        original: Original time series
        reconstructed: Reconstructed time series
        anomaly_mask: Boolean mask for high-error zones (if None, computed from error)
        title: Plot title
        error_threshold: Threshold for error zones (if None, uses 90th percentile)
        show_error_zones: If True, highlight error zones with red background
    """
    t = np.arange(len(original))
    
    # Compute reconstruction error
    error = np.abs(original - reconstructed)
    
    # Determine anomaly mask from error if not provided
    if anomaly_mask is None:
        if error_threshold is None:
            error_threshold = np.percentile(error, 90)
        anomaly_mask = error > error_threshold

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top panel: Original vs Reconstructed
    ax1 = axes[0]
    ax1.plot(t, original, label="Original", linewidth=2.5, color='blue', alpha=0.8, zorder=2)
    ax1.plot(t, reconstructed, label="Reconstruction", linewidth=2, 
            color='red', linestyle='--', alpha=0.8, zorder=2)
    
    # Highlight error zones with red background
    if show_error_zones and anomaly_mask.any():
        error_indices = np.where(anomaly_mask)[0]
        if len(error_indices) > 0:
            # Group consecutive errors
            zones = []
            start = error_indices[0]
            for i in range(1, len(error_indices)):
                if error_indices[i] != error_indices[i-1] + 1:
                    zones.append((start, error_indices[i-1]))
                    start = error_indices[i]
            zones.append((start, error_indices[-1]))
            
            # Draw red error zones
            for start_idx, end_idx in zones:
                ax1.axvspan(start_idx, end_idx, alpha=0.3, color='red', zorder=0,
                           label="High Error Zone" if start_idx == zones[0][0] else "")
    
    # Mark high-error points
    if anomaly_mask.any():
        ax1.scatter(t[anomaly_mask], np.array(original)[anomaly_mask],
                   marker="o", s=80, color='darkred', label="High Error Point",
                   zorder=5, edgecolors='black', linewidths=1.5)
    
    ax1.set_ylabel("Value", fontsize=12, fontweight='bold')
    if title:
        ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Reconstruction Error
    ax2 = axes[1]
    ax2.plot(t, error, label="Reconstruction Error", linewidth=2, 
            color='red', alpha=0.8, zorder=2)
    
    if error_threshold is not None:
        ax2.axhline(error_threshold, linestyle="--", linewidth=2, color='darkred',
                   label=f"Error Threshold ({error_threshold:.2f})", zorder=3)
        ax2.fill_between(t, error_threshold, error, where=(error > error_threshold),
                        alpha=0.3, color='red', zorder=1, label="Above Threshold")
    
    ax2.set_xlabel("Time Step", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Absolute Error", fontsize=12, fontweight='bold')
    ax2.set_title("Reconstruction Error Over Time", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, axes


def plot_temporal_segmentation(timestamps, values, segments, segment_labels=None, 
                               title=None, anomaly_segments=None):
    """
    Plot time series with temporal segmentation highlighted.
    
    Args:
        timestamps: Time values
        values: Series values
        segments: List of (start_idx, end_idx) tuples for segments
        segment_labels: Labels for each segment
        title: Plot title
        anomaly_segments: List of segment indices that are anomalies (highlighted in red)
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot full series
    ax.plot(timestamps, values, linewidth=2, color='blue', alpha=0.5, zorder=1)
    
    # Color map for segments
    colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
    
    # Plot segments
    for idx, (start_idx, end_idx) in enumerate(segments):
        if start_idx < len(timestamps) and end_idx < len(timestamps):
            segment_times = timestamps[start_idx:end_idx+1]
            segment_values = values[start_idx:end_idx+1]
            
            # Use red for anomaly segments
            if anomaly_segments is not None and idx in anomaly_segments:
                color = 'red'
                alpha = 0.6
                label = f"Anomaly Segment {idx+1}" if idx == anomaly_segments[0] else ""
            else:
                color = colors[idx]
                alpha = 0.8
                label = segment_labels[idx] if segment_labels else f"Segment {idx+1}"
            
            ax.plot(segment_times, segment_values, linewidth=3, color=color, 
                   alpha=alpha, label=label, zorder=2)
            
            # Add vertical lines at segment boundaries
            if idx > 0:
                ax.axvline(timestamps[start_idx], linestyle='--', 
                          color='gray', alpha=0.5, linewidth=1, zorder=0)
    
    ax.set_xlabel("Time", fontsize=12, fontweight='bold')
    ax.set_ylabel("Value", fontsize=12, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax

