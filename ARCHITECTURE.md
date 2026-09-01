# SatQuery System Architecture

## Overview
SatQuery is a multi-layer AI system for satellite image query and analysis, implementing USP-1 (Optical-SAR fusion) and USP-2 (Evidence-grounded reasoning).

## Architecture Layers

### 1. Presentation Layer (CLI)
**Location**: `/cli`

**Components**:
- `image_uploader.py` - Handles image selection (GeoTIFF, TIFF, PNG, JPEG)
- `query_console.py` - Text input for queries
- `results_viewer.py` - Visual & textual results display

**Functionality**: User interface for uploading satellite images and submitting queries.

---

### 2. API & Session Layer
**Location**: `/api`

**Components**:
- `gateway.py` - FastAPI gateway for all incoming requests
- `session_manager.py` - Manages user sessions and state across queries

**Endpoints**:
- `POST /api/v1/query` - Main query processing endpoint
- `GET /api/v1/session/{session_id}` - Session retrieval
- `GET /api/v1/health` - Health check
- `GET /api/v1/tools` - List available tools

---

### 3. Agentic Orchestration Layer
**Location**: `/agentic_layer`

**Components**:

#### 3.1 Input Validator
- Validates input format, quality, and geo-metadata
- Ensures images meet size and format requirements
- Parses and validates metadata

#### 3.2 Query Interpreter
- Classifies query into task types (VQA, grounding, change detection, SAR fusion)
- Extracts entities and intent
- Structures query for processing

#### 3.3 Tool Selector
- Selects appropriate specialist models based on task requirements
- Sequences tool execution order
- Picks from registry: VQA, grounding, change detection, SAR fusion

#### 3.4 Execution Engine
- Executes selected models with parameters
- Aggregates results from multiple tools
- Generates confidence scores and audit logs

---

### 4. Specialist Model Layer
**Location**: `/models`

**Models**:

#### 4.1 VQA Model (`vqa_model.py`)
- Single-image Q&A with scene description
- Based on BLIP architecture
- Handles: "What is", "How many", "Describe"

#### 4.2 Text-Guided Grounding Model (`grounding_model.py`)
- Localizes regions referred to in query
- Based on GroundingDINO
- Returns bounding boxes with confidence scores

#### 4.3 Change Detection Model (`change_detection_model.py`)
- Bi-temporal change VQA analysis
- Siamese network architecture
- Compares temporal images

#### 4.4 Optical-SAR Fusion Model (`sar_fusion_model.py`)
- Joint cross-modal information extraction
- Fuses optical and SAR data
- **USP-1**: Bypasses default grounding for SAR queries

---

### 5. Data & Benchmark Layer
**Location**: `/data_layer`

**Datasets**:
- **BigEarthNet** - Sentinel-1 SAR + Sentinel-2 optical training
- **VRSBench / RSVQA** - VQA evaluation & captioning
- **CDVQA** - Change-based VQA evaluation
- **ISRO SAC eval set** - SAR + optical fusion
- **CARTOSAT-2.5** - Optical with held-out test labels

---

## Data Flow

```
1. User uploads image + query (CLI)
   ↓
2. Request sent to API Gateway
   ↓
3. Input Validator checks format/quality/geo-metadata
   ↓
4. Query Interpreter classifies task & extracts entities
   ↓
5. Tool Selector picks specialist model(s)
   ↓
6. Execution Engine runs models & aggregates results
   ↓
7. Results returned with confidence + audit log
   ↓
8. Results Viewer displays output
```

---

## USP Implementation

### USP-1: Optical-SAR Fusion
- Paired upload skips default grounding
- Direct routing to SAR fusion model
- Joint feature extraction from both modalities

### USP-2: Evidence-Grounded Reasoning
- Full audit trail of reasoning
- Confidence scores per tool
- Visual evidence overlay
- Not just plain black-box output

---

## Configuration
**Location**: `/config`

- `settings.py` - Centralized configuration
- `.env` - Environment variables (not committed)
- `.env.example` - Template for environment setup

---

## Execution

### Start API Server:
```bash
python main.py
```

### Start CLI:
```bash
python cli.py
```

---

## Technology Stack

- **API Framework**: FastAPI + Uvicorn
- **ML Framework**: PyTorch + Transformers
- **Computer Vision**: OpenCV + Pillow
- **Geospatial**: GDAL + Rasterio + GeoPandas
- **Database**: SQLAlchemy + PostgreSQL + Redis (caching)

---

## Notes

This is a **working prototype** with placeholder model implementations. Production deployment would require:

1. Actual trained model weights
2. GPU infrastructure for inference
3. Proper database setup
4. Authentication & authorization
5. Load balancing & caching
6. Frontend interface (handled by frontend team)
