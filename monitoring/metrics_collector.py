import time
from typing import Dict, Any, List
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MetricsCollector:
    """
    Singleton Metrics Collector for API request latency, model invocations, error counts, and cache performance.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsCollector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.model_calls: Dict[str, int] = {
            'vqa_model': 0,
            'grounding_model': 0,
            'change_detection_model': 0,
            'sar_fusion_model': 0
        }
        self.latencies: List[float] = []
        self._initialized = True

    def record_request(self, latency_sec: float, success: bool = True, cached: bool = False):
        """Record API query execution metric"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        self.latencies.append(latency_sec)
        if len(self.latencies) > 500:
            self.latencies.pop(0)

    def record_model_call(self, tool_id: str):
        """Record specialist model execution count"""
        if tool_id in self.model_calls:
            self.model_calls[tool_id] += 1
        else:
            self.model_calls[tool_id] = 1

    def get_summary(self) -> Dict[str, Any]:
        """Return metric summary dictionary"""
        avg_latency = float(sum(self.latencies) / len(self.latencies)) if self.latencies else 0.0
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'avg_latency_sec': round(avg_latency, 4),
            'model_calls': self.model_calls
        }
