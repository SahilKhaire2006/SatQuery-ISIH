from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from pydantic import BaseModel
import uuid
from datetime import datetime

from api.session_manager import SessionManager
from agentic_layer.orchestrator import AgenticOrchestrator

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


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    image_format: Optional[str] = None
    geo_metadata: Optional[dict] = None


class QueryResponse(BaseModel):
    session_id: str
    query_id: str
    status: str
    results: dict
    confidence: float
    audit_log: dict
    timestamp: str


@app.get("/")
async def root():
    return {
        "service": "SatQuery API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(
    query: str = Form(...),
    image: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    geo_metadata: Optional[str] = Form(None)
):
    """
    Main endpoint for satellite image query processing
    """
    try:
        # Create or retrieve session
        if not session_id:
            session_id = str(uuid.uuid4())
            session_manager.create_session(session_id)
        
        # Read image data
        image_data = await image.read()
        
        # Process through orchestrator
        result = await orchestrator.process_request(
            session_id=session_id,
            query=query,
            image_data=image_data,
            image_filename=image.filename,
            geo_metadata=geo_metadata
        )
        
        # Update session
        session_manager.update_session(session_id, result)
        
        return QueryResponse(
            session_id=session_id,
            query_id=result['query_id'],
            status=result['status'],
            results=result['results'],
            confidence=result['confidence'],
            audit_log=result['audit_log'],
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/session/{session_id}")
async def get_session(session_id: str):
    """
    Retrieve session information
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "operational",
            "orchestrator": "operational",
            "models": "operational"
        }
    }


@app.get("/api/v1/tools")
async def list_tools():
    """
    List available specialist tools/models
    """
    return {
        "tools": orchestrator.get_available_tools(),
        "count": len(orchestrator.get_available_tools())
    }
