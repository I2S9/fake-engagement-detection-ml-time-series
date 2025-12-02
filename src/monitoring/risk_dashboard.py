"""
Risk ML Monitoring Dashboard Module.
Provides real-time monitoring, alerting, and dashboard metrics for the fraud detection system.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json


class RiskMonitor:
    """
    Risk monitoring system for fake engagement detection.
    Tracks metrics, generates alerts, and maintains dashboard statistics.
    """
    
    def __init__(self, alert_threshold: float = 0.7, alert_window_hours: int = 24):
        """
        Initialize risk monitor.
        
        Args:
            alert_threshold: Score threshold for generating alerts
            alert_window_hours: Time window for aggregating alerts
        """
        self.alert_threshold = alert_threshold
        self.alert_window_hours = alert_window_hours
        self.alerts_history = []
        self.metrics_history = []
    
    def check_alert(self, user_id: str, score: float, timestamp: datetime, 
                   metadata: Optional[Dict] = None) -> bool:
        """
        Check if a prediction score triggers an alert.
        
        Args:
            user_id: User identifier
            score: Anomaly score (0-1)
            timestamp: Prediction timestamp
            metadata: Additional metadata (attack_type, features, etc.)
        
        Returns:
            True if alert is triggered
        """
        is_alert = score >= self.alert_threshold
        
        if is_alert:
            alert = {
                "user_id": user_id,
                "score": float(score),
                "timestamp": timestamp.isoformat(),
                "severity": self._compute_severity(score),
                "metadata": metadata or {}
            }
            self.alerts_history.append(alert)
        
        return is_alert
    
    def _compute_severity(self, score: float) -> str:
        """Compute alert severity based on score."""
        if score >= 0.9:
            return "CRITICAL"
        elif score >= 0.8:
            return "HIGH"
        elif score >= 0.7:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_alert_summary(self, hours: Optional[int] = None) -> Dict:
        """
        Get summary of alerts in the specified time window.
        
        Args:
            hours: Number of hours to look back (None = all history)
        
        Returns:
            Dictionary with alert statistics
        """
        if hours is None:
            alerts = self.alerts_history
        else:
            cutoff = datetime.now() - timedelta(hours=hours)
            alerts = [
                a for a in self.alerts_history
                if datetime.fromisoformat(a["timestamp"]) >= cutoff
            ]
        
        if not alerts:
            return {
                "total_alerts": 0,
                "by_severity": {},
                "avg_score": 0.0,
                "top_users": []
            }
        
        severity_counts = {}
        scores = []
        user_counts = {}
        
        for alert in alerts:
            severity = alert["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            scores.append(alert["score"])
            user_id = alert["user_id"]
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
        
        top_users = sorted(
            user_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        return {
            "total_alerts": len(alerts),
            "by_severity": severity_counts,
            "avg_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "top_users": [{"user_id": u, "alert_count": c} for u, c in top_users]
        }
    
    def record_metrics(self, metrics: Dict, timestamp: Optional[datetime] = None):
        """
        Record model performance metrics.
        
        Args:
            metrics: Dictionary of metrics (auc, precision, recall, etc.)
            timestamp: Timestamp for metrics (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        record = {
            "timestamp": timestamp.isoformat(),
            **metrics
        }
        self.metrics_history.append(record)
    
    def get_metrics_trend(self, metric_name: str, hours: Optional[int] = None) -> List[Tuple[datetime, float]]:
        """
        Get trend of a specific metric over time.
        
        Args:
            metric_name: Name of metric to track
            hours: Number of hours to look back (None = all history)
        
        Returns:
            List of (timestamp, value) tuples
        """
        if hours is None:
            records = self.metrics_history
        else:
            cutoff = datetime.now() - timedelta(hours=hours)
            records = [
                r for r in self.metrics_history
                if datetime.fromisoformat(r["timestamp"]) >= cutoff
            ]
        
        trend = []
        for record in records:
            if metric_name in record:
                timestamp = datetime.fromisoformat(record["timestamp"])
                trend.append((timestamp, record[metric_name]))
        
        return sorted(trend, key=lambda x: x[0])
    
    def export_dashboard_data(self, output_path: Path):
        """
        Export dashboard data to JSON for visualization.
        
        Args:
            output_path: Path to save JSON file
        """
        dashboard_data = {
            "alerts": self.alerts_history[-1000:],  # Last 1000 alerts
            "metrics": self.metrics_history[-100:],  # Last 100 metric records
            "summary": self.get_alert_summary(hours=self.alert_window_hours),
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
    
    def generate_risk_report(self) -> str:
        """
        Generate a human-readable risk report.
        
        Returns:
            Formatted report string
        """
        summary = self.get_alert_summary(hours=self.alert_window_hours)
        
        report = f"""
================================================================================
RISK MONITORING REPORT - Fake Engagement Detection
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ALERT SUMMARY (Last {self.alert_window_hours} hours):
  Total Alerts: {summary['total_alerts']}
  Average Score: {summary['avg_score']:.4f}
  Maximum Score: {summary['max_score']:.4f}
  
  By Severity:
"""
        for severity, count in summary['by_severity'].items():
            report += f"    {severity}: {count}\n"
        
        if summary['top_users']:
            report += "\n  Top Alerted Users:\n"
            for user_info in summary['top_users'][:5]:
                report += f"    User {user_info['user_id']}: {user_info['alert_count']} alerts\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report

