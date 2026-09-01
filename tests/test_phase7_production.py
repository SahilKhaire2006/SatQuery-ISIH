import sys
import os
import asyncio
import cv2
import numpy as np
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.gateway import app
from api.auth import create_access_token, verify_token
from utils.cache import QueryCache
from models.model_manager import ModelManager
from monitoring.metrics_collector import MetricsCollector
from monitoring.prometheus_exporter import PrometheusExporter
from monitoring.alert_manager import AlertManager

client = TestClient(app)


def test_jwt_auth():
    token = create_access_token(data={"sub": "admin"})
    assert token is not None
    payload = verify_token(token)
    assert payload is not None
    assert payload.get("sub") == "admin"


def test_query_cache():
    cache = QueryCache()
    dummy_result = {'query_id': 'test_123', 'status': 'success', 'results': {'answer': 'cached'}}
    cache.set("test query", b"img_bytes", dummy_result)
    retrieved = cache.get("test query", b"img_bytes")
    assert retrieved is not None
    assert retrieved['query_id'] == 'test_123'


def test_model_manager_precision():
    manager = ModelManager()
    manager.clear_gpu_memory()
    assert hasattr(manager, 'use_fp16')


def test_monitoring_collector_and_exporter():
    collector = MetricsCollector()
    collector.record_request(0.12, success=True, cached=False)
    collector.record_model_call('vqa_model')
    summary = collector.get_summary()
    assert summary['total_requests'] > 0

    exporter = PrometheusExporter()
    metrics_text = exporter.generate_prometheus_metrics()
    assert "satquery_requests_total" in metrics_text
    assert "satquery_request_latency_seconds" in metrics_text

    alert_mgr = AlertManager()
    alerts = alert_mgr.check_alerts()
    assert isinstance(alerts, list)


def test_api_auth_endpoint():
    resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "secret_admin_password"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_api_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "satquery_requests_total" in resp.text


def test_api_batch_query_endpoint():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.png', img)

    resp = client.post(
        "/api/v1/batch-query",
        data={"queries": ["What is here?", "Locate buildings"]},
        files={"image": ("test.png", buf.tobytes(), "image/png")}
    )
    assert resp.status_code == 200
    assert resp.json()["batch_count"] == 2


if __name__ == '__main__':
    test_jwt_auth()
    test_query_cache()
    test_model_manager_precision()
    test_monitoring_collector_and_exporter()
    test_api_auth_endpoint()
    test_api_metrics_endpoint()
    test_api_batch_query_endpoint()
    print("All Phase 7 Production Optimization tests passed successfully!")
