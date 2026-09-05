# Model 2 — Integration Guide

> Step-by-step guide showing exactly which existing files to modify and what new files to create to integrate Model 2 into the SatQuery system.

---

## Pre-Requisites

### 1. Install New Dependencies

```bash
pip install sentinelhub earthaccess
```

Add to `requirements.txt`:
```
sentinelhub>=3.10.0
earthaccess>=0.9.0
```

### 2. Register for Copernicus Data Space

1. Go to https://dataspace.copernicus.eu/
2. Create free account
3. Create OAuth client → get `client_id` + `client_secret`

### 3. Update `.env`

```env
# === NEW: Copernicus Data Space (Sentinel Hub) ===
COPERNICUS_CLIENT_ID=your_client_id_here
COPERNICUS_CLIENT_SECRET=your_client_secret_here
SENTINEL_HUB_BASE_URL=https://sh.dataspace.copernicus.eu

# === NEW: Disaster Analysis Config ===
DISASTER_TEMPORAL_LOOKBACK_DAYS=14
FLOOD_NDWI_THRESHOLD=0.3
SAR_FLOOD_THRESHOLD=-15
```

---

## File-by-File Changes

### A. Config Layer

#### [MODIFY] `config/settings.py`

Add after line 41 (`HF_FEATURE_MODEL`):

```python
# Copernicus Data Space / Sentinel Hub
COPERNICUS_CLIENT_ID = os.getenv('COPERNICUS_CLIENT_ID', '')
COPERNICUS_CLIENT_SECRET = os.getenv('COPERNICUS_CLIENT_SECRET', '')
SENTINEL_HUB_BASE_URL = os.getenv('SENTINEL_HUB_BASE_URL', 
    'https://sh.dataspace.copernicus.eu')

# Disaster Analysis Settings
DISASTER_TEMPORAL_LOOKBACK_DAYS = int(os.getenv('DISASTER_TEMPORAL_LOOKBACK_DAYS', 14))
FLOOD_NDWI_THRESHOLD = float(os.getenv('FLOOD_NDWI_THRESHOLD', 0.3))
SAR_FLOOD_THRESHOLD = float(os.getenv('SAR_FLOOD_THRESHOLD', -15))
```

Add `'disaster_grounding_model'` to `AVAILABLE_TOOLS` list:

```python
AVAILABLE_TOOLS = [
    'vqa_model',
    'grounding_model',
    'building_detector',
    'roboflow_building_detector',
    'spectral_index_model',
    'change_detection_model',
    'sar_fusion_model',
    'disaster_grounding_model',   # ← NEW
]
```

---

### B. Geospatial Layer

#### [NEW] `geospatial/sentinel_fetcher.py`

Primary Sentinel Hub integration for fetching real Sentinel-2 and Sentinel-1 imagery.

**Key functions:**
- `fetch_sentinel2_rgb(lat, lon, date_range, area_meters)` → True color image
- `fetch_sentinel1_sar(lat, lon, date_range, area_meters)` → SAR backscatter
- `fetch_ndwi_layer(lat, lon, date_range, area_meters)` → NDWI water map
- `fetch_multi_temporal(lat, lon, dates_list, area_meters)` → Temporal series

#### [NEW] `geospatial/imagery_router.py`

Decides which imagery source to use based on disaster type + conditions.

**Key function:**
- `route_imagery_request(disaster_type, location, temporal_needed)` → imagery source decision

#### [EXISTING] `geospatial/map_fetcher.py`

No changes needed. Becomes the fallback automatically via `imagery_router.py`.

---

### C. Model Layer

#### [NEW] `models/disaster_analysis/__init__.py`

```python
"""
Disaster Analysis Models for Model 2 (Text-Guided Grounding)
Supports: Flood mapping, Earthquake assessment, Evacuation planning
"""

from models.disaster_analysis.disaster_grounding_model import DisasterGroundingModel
from models.disaster_analysis.flood_analyzer import FloodAnalyzer
from models.disaster_analysis.earthquake_analyzer import EarthquakeAnalyzer
from models.disaster_analysis.disaster_llm_reasoner import DisasterLLMReasoner
```

#### [NEW] `models/disaster_analysis/disaster_grounding_model.py`

The main Model 2 entry point. Implements the same `predict(image, query, parameters)` interface as all other models.

```python
class DisasterGroundingModel:
    """
    Model 2: Text-Guided Disaster Grounding
    
    Unlike other models, this one:
    1. Ignores the `image` parameter (fetches its own)
    2. Extracts location from the query
    3. Fetches real satellite imagery from Sentinel Hub / Esri
    4. Runs disaster-specific analysis pipeline
    5. Returns comprehensive disaster report
    """
    
    def predict(self, image, query, parameters) -> Dict:
        # Step 1: Extract location (geocode)
        # Step 2: Determine disaster type
        # Step 3: Fetch real satellite imagery
        # Step 4: Run flood/earthquake analyzer
        # Step 5: Run scene description (VQA on fetched image)
        # Step 6: LLM synthesis → disaster report
        # Step 7: Return results in standard format
```

#### [NEW] `models/disaster_analysis/flood_analyzer.py`

Flood-specific analysis: NDWI, SAR, progression, infrastructure impact, evacuation.

#### [NEW] `models/disaster_analysis/earthquake_analyzer.py`

Earthquake-specific analysis: change detection, damage zoning.

#### [NEW] `models/disaster_analysis/disaster_llm_reasoner.py`

Groq LLM integration for disaster intelligence synthesis.

---

### D. Agentic Orchestration Layer

#### [MODIFY] `agentic_layer/query_interpreter.py`

Add disaster intent classification to `_classify_intent()` method.

Insert **BEFORE** the existing `# Building detection intent` block (it should be checked first since disaster takes priority):

```python
def _classify_intent(self, query: str, interpretation: Dict) -> str:
    query_lower = query.lower()
    task_type = interpretation.get('task_type', 'vqa')
    entities = interpretation.get('entities', [])
    
    # ===== NEW: Disaster intent (highest priority) =====
    flood_keywords = ['flood', 'inundation', 'water level', 'submerged', 
                      'flood progression', 'flood extent', 'riverbank overflow']
    earthquake_keywords = ['earthquake', 'seismic', 'structural damage', 
                          'collapsed building', 'richter', 'magnitude']
    disaster_keywords = ['disaster', 'evacuation', 'tsunami', 'cyclone', 
                        'hurricane', 'landslide', 'wildfire', 'storm surge',
                        'rescue', 'damage assessment', 'relief', 'emergency response']
    
    if any(kw in query_lower for kw in flood_keywords):
        return 'disaster_flood'
    if any(kw in query_lower for kw in earthquake_keywords):
        return 'disaster_earthquake'
    if any(kw in query_lower for kw in disaster_keywords):
        return 'disaster_general'
    # ===== END NEW =====
    
    # ... existing building/water/vegetation/change/vqa logic unchanged ...
```

#### [MODIFY] `agentic_layer/tool_selector.py`

Add to `tool_registry` in `__init__`:

```python
'disaster_grounding_model': {
    'name': 'Disaster Management Grounding Model',
    'tasks': ['disaster_analysis', 'flood_mapping', 'earthquake_assessment',
              'evacuation_planning', 'disaster_prediction'],
    'priority': 1
}
```

Add intent routing in `select_tools()`, before the `else` (general_vqa) block:

```python
elif intent.startswith('disaster_'):
    disaster_type = intent.replace('disaster_', '')
    logger.info(f"Intent routing: {intent} -> disaster_grounding_model")
    tools = [{
        'tool_id': 'disaster_grounding_model',
        'tool_name': 'Disaster Management Grounding Model',
        'order': 1,
        'parameters': {
            'disaster_type': disaster_type,
            'location': interpretation.get('original_query', ''),
            'coordinates': interpretation.get('spatial_metadata', {}).get('coordinates'),
            'temporal_range': interpretation.get('temporal_aspects', {})
        },
        'rationale': f'Intent-based routing for {disaster_type} disaster analysis'
    }]
```

#### [MODIFY] `agentic_layer/execution_engine.py`

Add import at top:

```python
try:
    from models.disaster_analysis.disaster_grounding_model import DisasterGroundingModel
    DISASTER_MODEL_AVAILABLE = True
except ImportError:
    DISASTER_MODEL_AVAILABLE = False
```

Add to `self.models` dict in `__init__`:

```python
if DISASTER_MODEL_AVAILABLE:
    self.models['disaster_grounding_model'] = DisasterGroundingModel()
    logger.info("Disaster Grounding Model loaded successfully")
```

---

### E. Orchestrator — Special Handling

#### [MODIFY] `agentic_layer/orchestrator.py`

The orchestrator currently fetches a map tile when no image is uploaded (lines 86-118). For disaster queries, the `DisasterGroundingModel` handles its own imagery fetching, so the orchestrator should **skip** the default tile fetch when a disaster tool is selected.

Add a check after tool selection (around line 80):

```python
# Check if disaster model is selected (it fetches its own imagery)
is_disaster_query = any(
    t.get('tool_id') == 'disaster_grounding_model' 
    for t in selected_tools
)

if is_disaster_query:
    # Disaster model handles its own satellite imagery fetching
    # Pass a placeholder image; the model will ignore it
    logger.info("Disaster query detected — model will fetch its own real satellite imagery")
    if image_for_execution is None:
        image_for_execution = np.zeros((512, 512, 3), dtype=np.uint8)
```

---

### F. Frontend

#### [NEW] `frontend/src/components/DisasterAnalysisCard.jsx`

Display disaster analysis results including:
- Flood extent statistics
- Severity badge (ADVISORY / WARNING / EMERGENCY / CATASTROPHIC)
- Evacuation zone map
- Temporal progression timeline
- Predictions
- Infrastructure impact count

#### [MODIFY] `frontend/src/components/VisualEvidenceViewer.jsx`

Add support for rendering:
- Flood mask overlay (blue tint on affected areas)
- Evacuation zone overlay (green=safe, yellow=warning, red=danger)
- Temporal progression carousel (before → during → current)
- Damage heatmap for earthquake analysis

---

## Testing Checklist

### Smoke Tests

```bash
# Test 1: Flood query (text only, no image)
curl -X POST http://localhost:8000/api/v1/query \
  -F "query=Show flood extent in Wayanad Kerala" \
  -F "image=@empty_placeholder.png"

# Test 2: Earthquake query
curl -X POST http://localhost:8000/api/v1/query \
  -F "query=Assess earthquake damage in Turkey Hatay province" \
  -F "image=@empty_placeholder.png"

# Test 3: Verify Model 1 still works (regression)
curl -X POST http://localhost:8000/api/v1/query \
  -F "query=How many buildings are in this image" \
  -F "image=@satelite-img.png"
```

### Verification

- [ ] Disaster query returns real satellite imagery (check `imagery_source` in response)
- [ ] Flood analysis metrics are computed (not zeros or placeholders)
- [ ] LLM report references actual computed numbers
- [ ] Model 1 queries (building/water/vegetation) still work correctly
- [ ] Audit trail includes full disaster pipeline steps
- [ ] Response time < 30 seconds

---

## Rollback Plan

If Model 2 integration causes issues with existing functionality:

1. Remove `'disaster_grounding_model'` from `AVAILABLE_TOOLS` in `config/settings.py`
2. Remove the disaster intent classification block from `query_interpreter.py`
3. Remove the disaster routing block from `tool_selector.py`
4. Remove the disaster model registration from `execution_engine.py`

This completely disables Model 2 without affecting any Model 1 functionality.
