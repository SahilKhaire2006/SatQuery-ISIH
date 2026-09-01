import pytest
from fastapi.testclient import TestClient
import main
from api.gateway import app

client = TestClient(app)


def test_main_uses_configured_port(monkeypatch):
    """The app should start on the configured API port, not a hard-coded value."""
    captured = {}

    def fake_run(app_name, host, port, reload, log_level):
        captured["app_name"] = app_name
        captured["host"] = host
        captured["port"] = port
        captured["reload"] = reload
        captured["log_level"] = log_level

    monkeypatch.setattr(main.uvicorn, "run", fake_run)
    main.main()

    assert captured["app_name"] == "api.gateway:app"
    assert captured["host"] == main.API_HOST
    assert captured["port"] == main.API_PORT
    assert captured["reload"] is False
    assert captured["log_level"] == "debug"


def test_vqa_model_produces_land_use_assessment():
    """The VQA model should assess imagery instead of echoing the input query."""
    from models.vqa_model import VQAModel

    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:, :, 0] = 140
    image[:, :, 1] = 170
    image[:, :, 2] = 120

    result = VQAModel().predict(image, "Can I use this land for a small-scale warehouse?")

    assert result["output"]["answer"]
    assert "warehouse" in result["output"]["answer"].lower()
    assert "Based on the satellite image" not in result["output"]["answer"]


def test_vqa_model_uses_requested_use_university():
    """The VQA model should answer using the actual requested development type, not a warehouse template."""
    from models.vqa_model import VQAModel

    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:, :, 0] = 140
    image[:, :, 1] = 170
    image[:, :, 2] = 120

    result = VQAModel().predict(image, "Can a university be built in this land?")

    answer = result["output"]["answer"].lower()
    assert "university" in answer
    assert "small-scale warehouse" not in answer


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
