from typing import List, Dict, Any
from monitoring.metrics_collector import MetricsCollector
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AlertManager:
    """
    Monitors system metrics and triggers health alerts when latency or error thresholds are breached.
    """

    def __init__(self):
        self.collector = MetricsCollector()
        self.max_latency_threshold_sec = 5.0
        self.max_error_rate_threshold = 0.20

    def check_alerts(self) -> List[Dict[str, Any]]:
        """
        Check metric thresholds and return list of active alerts.
        """
        alerts = []
        summary = self.collector.get_summary()

        total = summary.get('total_requests', 0)
        failed = summary.get('failed_requests', 0)
        avg_lat = summary.get('avg_latency_sec', 0.0)

        if avg_lat > self.max_latency_threshold_sec:
            alerts.append({
                'level': 'WARNING',
                'alert': 'HighLatency',
                'message': f"Average query latency ({avg_lat:.2f}s) exceeds threshold ({self.max_latency_threshold_sec}s)."
            })

        if total >= 5:
            error_rate = failed / float(total)
            if error_rate > self.max_error_rate_threshold:
                alerts.append({
                    'level': 'CRITICAL',
                    'alert': 'HighErrorRate',
                    'message': f"Query error rate ({error_rate:.1%}) exceeds threshold ({self.max_error_rate_threshold:.1%})."
                })

        return alerts
