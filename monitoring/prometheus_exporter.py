from monitoring.metrics_collector import MetricsCollector


class PrometheusExporter:
    """
    Exports Prometheus-formatted text metrics for system scraping.
    """

    def __init__(self):
        self.collector = MetricsCollector()

    def generate_prometheus_metrics(self) -> str:
        """
        Format metrics into Prometheus exposition format text.
        """
        summary = self.collector.get_summary()

        lines = [
            "# HELP satquery_requests_total Total number of processed query requests.",
            "# TYPE satquery_requests_total counter",
            f"satquery_requests_total {summary['total_requests']}",
            "",
            "# HELP satquery_requests_successful Successful query count.",
            "# TYPE satquery_requests_successful counter",
            f"satquery_requests_successful {summary['successful_requests']}",
            "",
            "# HELP satquery_requests_failed Failed query count.",
            "# TYPE satquery_requests_failed counter",
            f"satquery_requests_failed {summary['failed_requests']}",
            "",
            "# HELP satquery_request_latency_seconds Average request latency in seconds.",
            "# TYPE satquery_request_latency_seconds gauge",
            f"satquery_request_latency_seconds {summary['avg_latency_sec']}",
            "",
            "# HELP satquery_cache_hits_total Total query cache hits.",
            "# TYPE satquery_cache_hits_total counter",
            f"satquery_cache_hits_total {summary['cache_hits']}",
            "",
            "# HELP satquery_model_calls_total Total model invocations by tool ID.",
            "# TYPE satquery_model_calls_total counter"
        ]

        for tool_id, count in summary.get('model_calls', {}).items():
            lines.append(f'satquery_model_calls_total{{model="{tool_id}"}} {count}')

        lines.append("")
        return "\n".join(lines)
