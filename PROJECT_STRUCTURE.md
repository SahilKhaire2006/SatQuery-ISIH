# SatQuery Project Structure

## Directory Layout

```
SatQuery/
│
├── api/                          # API & Session Layer
│   ├── __init__.py
│   ├── gateway.py                # FastAPI gateway with endpoints
│   └── session_manager.py        # Session state management
│
├── agentic_layer/                # Agentic Orchestration Layer
│   ├── __init__.py
│   ├── orchestrator.py           # Main orchestrator coordinating all components
│   ├── input_validator.py        # Input validation (format, quality, geo-metadata)
│   ├── query_interpreter.py      # Query interpretation & task classification
│   ├── tool_selector.py          # Tool selection & sequencing
│   └── execution_engine.py       # Model execution & result aggregation
│
├── models/                       # Specialist Model Layer
│   ├── __init__.py
│   ├── vqa_model.py              # VQA model (single-image Q&A)
│   ├── grounding_model.py        # Text-guided grounding (localization)
│   ├── change_detection_model.py # Bi-temporal change detection
│   └── sar_fusion_model.py       # Optical-SAR fusion model (USP-1)
│
├── cli/                          # Presentation Layer (CLI)
│   ├── __init__.py
│   ├── image_uploader.py         # Image selection & upload
│   ├── query_console.py          # Query text input
│   └── results_viewer.py         # Results display & export
│
├── data_layer/                   # Data & Benchmark Layer
│   ├── __init__.py
│   └── dataset_manager.py        # Dataset management (BigEarthNet, RSVQA, etc.)
│
├── config/                       # Configuration
│   ├── __init__.py
│   └── settings.py               # Centralized settings
│
├── utils/                        # Utilities
│   ├── __init__.py
│   └── logger.py                 # Logging utilities
│
├── tests/                        # Tests
│   ├── __init__.py
│   └── test_api.py               # API endpoint tests
│
├── scripts/                      # Setup scripts
│   ├── setup.sh                  # Linux/Mac setup
│   └── setup.bat                 # Windows setup
│
├── main.py                       # API server entry point
├── cli.py                        # CLI entry point
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Docker services orchestration
├── README.md                     # Project overview
├── ARCHITECTURE.md               # Detailed architecture documentation
├── API_USAGE.md                  # API usage guide
└── PROJECT_STRUCTURE.md          # This file
```

## Key Files Description

### Entry Points
- **main.py** - Starts the FastAPI server (default: http://localhost:8000)
- **cli.py** - Interactive command-line interface

### Core API
- **api/gateway.py** - REST API endpoints for query processing
- **api/session_manager.py** - Manages user sessions across multiple queries

### Orchestration Logic
- **agentic_layer/orchestrator.py** - Coordinates the entire processing pipeline
- **agentic_layer/input_validator.py** - Validates images, queries, and metadata
- **agentic_layer/query_interpreter.py** - Parses and classifies queries
- **agentic_layer/tool_selector.py** - Selects appropriate AI models
- **agentic_layer/execution_engine.py** - Executes models and aggregates results

### AI Models (Placeholders)
- **models/vqa_model.py** - Visual question answering (BLIP-based)
- **models/grounding_model.py** - Object localization (GroundingDINO-based)
- **models/change_detection_model.py** - Temporal change analysis (Siamese)
- **models/sar_fusion_model.py** - Cross-modal SAR-optical fusion

### CLI Components
- **cli/image_uploader.py** - Handles image file selection
- **cli/query_console.py** - Command-line query input
- **cli/results_viewer.py** - Formatted results display

### Configuration
- **config/settings.py** - All configurable parameters
- **.env.example** - Template for environment variables

## Quick Start

### 1. Setup (Windows)
```cmd
scripts\setup.bat
```

### 2. Setup (Linux/Mac)
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 3. Start API Server
```bash
python main.py
```

### 4. Use CLI Interface
```bash
python cli.py
```

### 5. Docker Deployment
```bash
docker-compose up -d
```

## Architecture Layers

1. **Presentation Layer** - CLI for user interaction (`cli/`)
2. **API & Session Layer** - REST API and state management (`api/`)
3. **Agentic Orchestration Layer** - Intelligence & coordination (`agentic_layer/`)
4. **Specialist Model Layer** - AI models for specific tasks (`models/`)
5. **Data & Benchmark Layer** - Dataset management (`data_layer/`)

## Implementation Notes

- All AI models are **placeholder implementations** in this prototype
- The system is fully functional for testing API flow and architecture
- Production deployment would require:
  - Real trained model weights
  - GPU infrastructure
  - Database setup (PostgreSQL + Redis)
  - Authentication system
  - Frontend interface (handled by frontend team)

## API Endpoints

- `POST /api/v1/query` - Process satellite image query
- `GET /api/v1/session/{id}` - Retrieve session info
- `GET /api/v1/health` - Health check
- `GET /api/v1/tools` - List available models
- `GET /docs` - Interactive API documentation (Swagger UI)

## Technology Stack

- **Backend**: FastAPI, Uvicorn
- **ML**: PyTorch, Transformers, OpenCV
- **Geospatial**: GDAL, Rasterio, GeoPandas
- **Database**: PostgreSQL, SQLAlchemy, Redis
- **Container**: Docker, Docker Compose

## Next Steps for Production

1. Integrate actual trained model weights
2. Set up GPU inference servers
3. Implement authentication & authorization
4. Deploy database and caching layer
5. Add monitoring and logging (Prometheus, Grafana)
6. Configure load balancing (Nginx)
7. Integrate with frontend application

---

**For detailed information**:
- Architecture: See `ARCHITECTURE.md`
- API Usage: See `API_USAGE.md`
- Overview: See `README.md`
