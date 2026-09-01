# SatQuery API Usage Guide

## Getting Started

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start API server
python main.py
```

Server will run at: `http://localhost:8000`

API docs available at: `http://localhost:8000/docs`

---

## API Endpoints

### 1. Process Query
**POST** `/api/v1/query`

Upload an image and submit a query for analysis.

**Request**:
- Method: `POST`
- Content-Type: `multipart/form-data`

**Form Data**:
- `query` (required): Natural language query string
- `image` (required): Image file (GeoTIFF, TIFF, PNG, JPEG)
- `session_id` (optional): Session ID for maintaining context
- `geo_metadata` (optional): JSON string with geo-metadata

**Example using cURL**:
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -F "query=How many buildings are visible in this image?" \
  -F "image=@/path/to/satellite_image.tif"
```

**Example using Python**:
```python
import requests

url = "http://localhost:8000/api/v1/query"

files = {
    'image': open('satellite_image.tif', 'rb')
}

data = {
    'query': 'How many buildings are visible in this image?'
}

response = requests.post(url, files=files, data=data)
result = response.json()
print(result)
```

**Response**:
```json
{
  "session_id": "uuid-string",
  "query_id": "uuid-string",
  "status": "success",
  "results": {
    "answer": "There are approximately 5-7 buildings visible in the image",
    "visual_output": null,
    "confidence_scores": {
      "vqa_model": 0.85
    },
    "details": {}
  },
  "confidence": 0.85,
  "audit_log": {
    "validation": {},
    "interpretation": {},
    "selected_tools": [],
    "execution": []
  },
  "timestamp": "2026-09-01T10:30:00"
}
```

---

### 2. Get Session
**GET** `/api/v1/session/{session_id}`

Retrieve information about a session.

**Example**:
```bash
curl -X GET "http://localhost:8000/api/v1/session/your-session-id"
```

---

### 3. Health Check
**GET** `/api/v1/health`

Check API server health status.

**Example**:
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-09-01T10:30:00",
  "services": {
    "api": "operational",
    "orchestrator": "operational",
    "models": "operational"
  }
}
```

---

### 4. List Tools
**GET** `/api/v1/tools`

List all available specialist tools/models.

**Example**:
```bash
curl -X GET "http://localhost:8000/api/v1/tools"
```

**Response**:
```json
{
  "tools": [
    "vqa_model",
    "grounding_model",
    "change_detection_model",
    "sar_fusion_model"
  ],
  "count": 4
}
```

---

## Query Examples

### VQA (Visual Question Answering)
```
Query: "What type of land cover is dominant in this image?"
Query: "How many roads intersect in this area?"
Query: "Describe the urban density in this satellite image"
```

### Grounding (Object Localization)
```
Query: "Where are the buildings located?"
Query: "Find water bodies in this image"
Query: "Locate agricultural fields"
```

### Change Detection
```
Query: "What changes occurred between these two time periods?"
Query: "Compare urban growth in this area"
Query: "Detect deforestation changes"
```

### SAR Fusion
```
Query: "Analyze this area using SAR and optical data"
Query: "Perform cross-modal fusion analysis"
Query: "Extract features from radar and optical imagery"
```

---

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Unique session identifier |
| `query_id` | string | Unique query identifier |
| `status` | string | Processing status (success/failed/error) |
| `results.answer` | string | Main answer to query |
| `results.confidence_scores` | object | Confidence per tool |
| `confidence` | float | Overall confidence (0.0-1.0) |
| `audit_log` | object | Full processing trail |
| `timestamp` | string | ISO 8601 timestamp |

---

## Error Handling

**Error Response**:
```json
{
  "detail": "Error message description"
}
```

**Common Status Codes**:
- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Not Found (session doesn't exist)
- `500` - Internal Server Error

---

## CLI Usage

For interactive CLI usage:

```bash
python cli.py
```

The CLI provides:
1. Interactive menu
2. Image file selection
3. Query input
4. Formatted results display
5. Export to JSON

---

## Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f api

# Stop services
docker-compose down
```

---

## Notes

- Maximum image size: 1024px (configurable in `.env`)
- Supported formats: GeoTIFF, TIFF, PNG, JPEG
- Session state persists in memory (use Redis for production)
- Models are placeholders in prototype (need actual weights for production)
