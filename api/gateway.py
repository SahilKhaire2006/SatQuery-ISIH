import time
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.session_manager import SessionManager
from api.auth import create_access_token, verify_token, Token
from agentic_layer.orchestrator import AgenticOrchestrator
from utils.cache import QueryCache
from monitoring.metrics_collector import MetricsCollector
from monitoring.prometheus_exporter import PrometheusExporter
from monitoring.alert_manager import AlertManager

app = FastAPI(title="SatQuery API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
session_manager = SessionManager()
orchestrator = AgenticOrchestrator()
cache = QueryCache()
metrics_collector = MetricsCollector()
prometheus_exporter = PrometheusExporter()
alert_manager = AlertManager()


class QueryResponse(BaseModel):
    session_id: str
    query_id: str
    status: str
    explanation: Optional[str] = None
    results: dict
    confidence: float
    audit_log: dict
    timestamp: str


class LoginRequest(BaseModel):
    username: str
    password: str


@app.get("/")
async def root():
    return {
        "service": "SatQuery API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.post("/api/v1/auth/token", response_model=Token)
@app.post("/api/v1/auth/login", response_model=Token)
async def login_for_access_token(login: LoginRequest):
    """
    Authenticate user and issue JWT access token
    """
    if login.username == "admin" and login.password == "secret_admin_password":
        access_token = create_access_token(data={"sub": login.username})
        return Token(access_token=access_token, token_type="bearer")
    raise HTTPException(status_code=401, detail="Incorrect username or password")


@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(
    query: str = Form(...),
    image: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    geo_metadata: Optional[str] = Form(None)
):
    """
    Main endpoint for satellite image query processing with caching & metric tracking
    """
    start_time = time.time()
    try:
        if not session_id:
            session_id = str(uuid.uuid4())
            session_manager.create_session(session_id)

        image_data = await image.read()

        # Check Cache
        cached_result = cache.get(query, image_data)
        if cached_result:
            metrics_collector.record_request(time.time() - start_time, success=True, cached=True)
            return QueryResponse(
                session_id=session_id,
                query_id=cached_result['query_id'],
                status=cached_result['status'],
                explanation=cached_result.get('explanation'),
                results=cached_result['results'],
                confidence=cached_result['confidence'],
                audit_log=cached_result['audit_log'],
                timestamp=datetime.now().isoformat()
            )

        # Process request via orchestrator
        result = await orchestrator.process_request(
            session_id=session_id,
            query=query,
            image_data=image_data,
            image_filename=image.filename,
            geo_metadata=geo_metadata
        )

        from utils.json_sanitizer import sanitize_for_json
        result = sanitize_for_json(result)

        session_manager.update_session(session_id, result)
        cache.set(query, image_data, result)

        metrics_collector.record_request(time.time() - start_time, success=True, cached=False)

        return QueryResponse(
            session_id=session_id,
            query_id=result['query_id'],
            status=result['status'],
            explanation=result.get('explanation'),
            results=result['results'],
            confidence=float(result['confidence']),
            audit_log=result['audit_log'],
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        metrics_collector.record_request(time.time() - start_time, success=False, cached=False)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/batch-query")
async def batch_process_queries(
    queries: List[str] = Form(...),
    image: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Batch processing endpoint for evaluating multiple queries over a satellite image
    """
    if not session_id:
        session_id = str(uuid.uuid4())
        session_manager.create_session(session_id)

    image_bytes = await image.read()
    results = []

    for q in queries:
        res = await orchestrator.process_request(
            session_id=session_id,
            query=q,
            image_data=image_bytes,
            image_filename=image.filename
        )
        results.append(res)

    return {
        "session_id": session_id,
        "batch_count": len(results),
        "results": results
    }


@app.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics exposition endpoint for scraping
    """
    body = prometheus_exporter.generate_prometheus_metrics()
    return Response(content=body, media_type="text/plain")


@app.get("/api/v1/session/{session_id}")
async def get_session(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.get("/api/v1/health")
async def health_check():
    alerts = alert_manager.check_alerts()
    return {
        "status": "healthy" if not alerts else "degraded",
        "timestamp": datetime.now().isoformat(),
        "alerts": alerts,
        "services": {
            "api": "operational",
            "orchestrator": "operational",
            "models": "operational",
            "cache": "operational"
        }
    }


@app.get("/api/v1/tools")
async def list_tools():
    return {
        "tools": orchestrator.get_available_tools(),
        "count": len(orchestrator.get_available_tools())
    }
