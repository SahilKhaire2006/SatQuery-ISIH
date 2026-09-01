import pytest
from fastapi.testclient import TestClient
from api.gateway import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_list_tools():
    """Test tools listing endpoint"""
    response = client.get("/api/v1/tools")
    assert response.status_code == 200
    assert "tools" in response.json()
    assert len(response.json()["tools"]) > 0
