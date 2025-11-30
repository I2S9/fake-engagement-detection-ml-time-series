# Fake Engagement Detection via Temporal ML

## Project Overview

This project aims to detect fake or inauthentic engagement behavior (likes, comments, views, shares) on social media platforms using time series modeling and anomaly detection techniques.

## Problem Definition and Design

### Analysis Level

**Primary focus: Video-level time series**

We analyze engagement patterns at the video level, where each time series represents the temporal evolution of engagement metrics (likes, comments, views, shares) for a single video over time.

**Rationale:**
- Fake engagement patterns are typically tied to specific videos (e.g., sudden bursts after publication)
- Video-level detection is more actionable for content moderation
- Time series are more standardized and comparable across videos
- Easier to interpret and validate results

**Secondary consideration: User-level analysis**

While the primary focus is video-level, the same methodology can be applied to user-level time series to detect accounts that consistently generate fake engagement across multiple videos.

### Definition of Fake Engagement

Fake engagement is characterized by temporal patterns that deviate significantly from authentic engagement behavior. We identify the following key patterns:

1. **Sudden impossible spikes**
   - Abrupt increases in engagement metrics that are physically impossible (e.g., 10,000 likes in 1 minute)
   - Magnitude of change exceeds realistic thresholds based on historical patterns

2. **Regular bursts during off-peak hours**
   - Systematic bursts of engagement during unusual time periods (e.g., consistent spikes at 3 AM)
   - Patterns that do not align with typical user behavior or timezone distributions

3. **Disconnected engagement from historical context**
   - Engagement that does not follow expected decay patterns after initial publication
   - Metrics that are inconsistent with video characteristics (e.g., high views but zero comments)
   - Sudden reversals or anomalies that cannot be explained by organic factors

4. **Suspicious correlation patterns**
   - Perfect or near-perfect correlation between different engagement metrics (unrealistic in organic behavior)
   - Synchronized spikes across multiple metrics without natural variation

5. **Temporal inconsistency**
   - Engagement patterns that violate expected temporal dependencies
   - Reconstruction errors from time series models that indicate anomalous sequences

### Success Metrics

We measure model performance using the following metrics:

**Primary metrics:**
- **AUC-ROC**: Area under the receiver operating characteristic curve
  - Target: AUC > 0.85 for distinguishing fake from authentic engagement
- **Precision**: Proportion of predicted fake cases that are actually fake
  - Target: Precision > 0.80 to minimize false positives
- **Recall**: Proportion of actual fake cases that are correctly identified
  - Target: Recall > 0.75 to catch most fake engagement
- **F1-Score**: Harmonic mean of precision and recall
  - Target: F1 > 0.77 (balanced performance)

**Secondary metrics:**
- **False Positive Rate (FPR)**: Rate of authentic engagement incorrectly flagged as fake
  - Target: FPR < 0.10 to avoid over-flagging legitimate content
- **Anomaly score distributions**: Separation between scores for fake vs authentic engagement
  - Target: Clear separation with minimal overlap in score distributions
- **Temporal precision**: Ability to identify the exact time windows where fake engagement occurs
  - Target: Localize anomalies within 1-hour windows

**Evaluation approach:**
- Train/validation/test split: 60/20/20 with explicit random seed for reproducibility
- Time-aware splitting: Ensure temporal ordering is preserved (no data leakage)
- Stratified sampling: Maintain class balance across splits
- Cross-validation: Use time series cross-validation for robust evaluation

### Model Objectives

The system should achieve:
- Real-time or near-real-time detection capability (inference < 1 second per video)
- Interpretable anomaly scores that can be explained to stakeholders
- Robust performance across different video categories and engagement scales
- Low computational cost for production deployment

## Architecture Overview

The project follows a modular ML pipeline structure:

```
project-root/
    README.md
    data/
        raw/
        processed/
    notebooks/
        01_exploration.ipynb
        02_feature_engineering.ipynb
        03_modeling.ipynb
        04_evaluation.ipynb
    src/
        data/
            load_data.py
            preprocess.py
            simulate_timeseries.py
        features/
            temporal_features.py
        models/
            lstm.py
            tcn.py
            autoencoder.py
        training/
            train.py
            evaluate.py
        inference/
            api.py
            inference_pipeline.py
        utils/
            logger.py
            config.py
    config/
        config.yaml
    deployment/
        Dockerfile
        requirements.txt
        fastapi_server.py
    tests/
        test_data.py
        test_features.py
        test_models.py
        test_training.py
```

## Installation

(To be completed)

## Usage

(To be completed)

## Limitations and Future Improvements

(To be completed)

