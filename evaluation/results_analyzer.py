from typing import Dict, Any


class BenchmarkResultsAnalyzer:
    """
    Analyzes and formats benchmark performance reports for SatQuery.
    """

    def summarize_report(self, report: Dict[str, Any]) -> str:
        """Format benchmark report into clean markdown summary"""
        dataset = report.get('dataset', 'Unknown')
        split = report.get('split', 'test')
        samples = report.get('samples_evaluated', 0)
        metrics = report.get('metrics', {})
        latency = report.get('avg_latency_seconds', 0.0)

        lines = [
            f"### 📊 Benchmark Performance Summary: {dataset} ({split} split)\n",
            f"- **Evaluated Samples**: {samples}",
            f"- **Average Inference Latency**: {latency:.4f}s per query\n",
            "#### Task Performance Metrics:"
        ]

        for metric_name, score in metrics.items():
            lines.append(f"  - **{metric_name.replace('_', ' ').title()}**: {score}")

        return "\n".join(lines)
