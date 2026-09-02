# Multi-Tool Architecture Implementation Tasks

## Phase 1: Foundation & Validation

### Task 1.1: Validate Intent Classification ✅ IMPLEMENTED
**Status:** Needs Testing  
**Priority:** HIGH  
**Estimated Time:** 15 minutes

**Description:** Test that query interpreter correctly classifies intents.

**Acceptance Criteria:**
- [ ] Query "count buildings" → `building_detection`
- [ ] Query "locate water bodies" → `water_detection`
- [ ] Query "analyze vegetation" → `vegetation_detection`
- [ ] Query "what changed" → `change_detection`
- [ ] Query "what is in this image" → `general_vqa`
- [ ] Intent logged before tool routing

**Files to Test:**
- `agentic_layer/query_interpreter.py`

**Test Command:**
```bash
# Start server and test with queries
python main.py
# Then send test queries via API or CLI
```

---

### Task 1.2: Validate Tool Routing ✅ IMPLEMENTED
**Status:** Needs Testing  
**Priority:** HIGH  
**Estimated Time:** 15 minutes

**Description:** Verify tool_selector routes to correct tools based on intent.

**Acceptance Criteria:**
- [ ] `building_detection` intent → `building_detector` tool
- [ ] `water_detection` intent → `spectral_index_model` (NDWI) tool
- [ ] `vegetation_detection` intent → `spectral_index_model` (NDVI) tool
- [ ] `change_detection` intent → `change_detection_model` tool
- [ ] `general_vqa` intent → `vqa_model` tool
- [ ] Logs show: "Intent routing: {intent} → {tool_id}"
- [ ] No hardcoded overrides; LLM's intent is respected

**Files to Test:**
- `agentic_layer/tool_selector.py`

**Test Command:**
```bash
# Check logs when processing queries
tail -f logs/satquery.log
```

---

## Phase 2: Standardized Output Schema

### Task 2.1: Define Schema Module
**Status:** Not Started  
**Priority:** CRITICAL  
**Estimated Time:** 30 minutes

**Description:** Create a standardized schema class/dataclass that all tools must return.

**Deliverables:**
- [ ] Create `models/tool_output_schema.py`
- [ ] Define `ToolOutput` dataclass with fields:
  - `type: Literal["detection", "segmentation", "vqa"]`
  - `detections: List[Detection]` (Detection = label, confidence, box)
  - `mask: Optional[np.ndarray]`
  - `answer_text: Optional[str]`
  - `status: Literal["ok", "no_detections", "not_applicable", "failed"]`
  - `model_name: str`
  - `execution_time: float`
- [ ] Add validation methods
- [ ] Add serialization to dict/JSON

**Files to Create:**
- `models/tool_output_schema.py`

**Example Structure:**
```python
from dataclasses import dataclass
from typing import List, Optional, Literal
import numpy as np

@dataclass
class Detection:
    label: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]

@dataclass
class ToolOutput:
    type: Literal["detection", "segmentation", "vqa"]
    detections: List[Detection]
    mask: Optional[np.ndarray]
    answer_text: Optional[str]
    status: Literal["ok", "no_detections", "not_applicable", "failed"]
    model_name: str
    execution_time: float
    
    def to_dict(self) -> dict:
        # Serialize to JSON-compatible dict
        pass
```

---

### Task 2.2: Refactor VQA Model to Use Schema
**Status:** Not Started  
**Priority:** HIGH  
**Estimated Time:** 20 minutes

**Description:** Update `vqa_model.py` to return standardized `ToolOutput`.

**Acceptance Criteria:**
- [ ] `predict()` method returns `ToolOutput` instance
- [ ] `type = "vqa"`
- [ ] `answer_text` contains VQA model's answer
- [ ] `detections = []` (VQA doesn't detect objects)
- [ ] `status = "ok"` or `"failed"`
- [ ] `execution_time` measured

**Files to Modify:**
- `models/vqa_model.py`

---

### Task 2.3: Refactor Grounding Model to Use Schema
**Status:** Not Started  
**Priority:** HIGH  
**Estimated Time:** 20 minutes

**Description:** Update `grounding_model.py` to return standardized `ToolOutput`.

**Acceptance Criteria:**
- [ ] `predict()` method returns `ToolOutput` instance
- [ ] `type = "detection"`
- [ ] `detections` list populated with Detection objects
- [ ] `status = "ok"` if detections found, else `"no_detections"`
- [ ] `execution_time` measured

**Files to Modify:**
- `models/grounding_model.py`

---

### Task 2.4: Refactor Building Detector to Use Schema
**Status:** Not Started  
**Priority:** HIGH  
**Estimated Time:** 20 minutes

**Description:** Update `building_detector.py` to return standardized `ToolOutput`.

**Acceptance Criteria:**
- [ ] `predict()` method returns `ToolOutput` instance
- [ ] `type = "detection"` or `"segmentation"` (if using U-Net mask)
- [ ] `detections` list populated with building detections
- [ ] `mask` field populated if segmentation used
- [ ] `status = "ok"`, `"no_detections"`, or `"failed"`
- [ ] `execution_time` measured

**Files to Modify:**
- `models/building_detector.py`

---

### Task 2.5: Update Execution Engine to Handle Schema
**Status:** Not Started  
**Priority:** HIGH  
**Estimated Time:** 30 minutes

**Description:** Update execution engine to work with standardized `ToolOutput`.

**Acceptance Criteria:**
- [ ] Extract results from `ToolOutput.to_dict()`
- [ ] Handle all status types appropriately
- [ ] Aggregate results uniformly without tool-specific logic
- [ ] Pass standardized data to result aggregator

**Files to Modify:**
- `agentic_layer/execution_engine.py`

---

## Phase 3: Roboflow Building Detection

### Task 3.1: Install Roboflow SDK
**Status:** Not Started  
**Priority:** HIGH  
**Estimated Time:** 5 minutes

**Description:** Add Roboflow inference-sdk to project dependencies.

**Acceptance Criteria:**
- [ ] Add `inference-sdk` to `requirements.txt`
- [ ] Install package: `pip install inference-sdk`
- [ ] Verify import: `from inference_sdk import InferenceHTTPClient`

**Files to Modify:**
- `requirements.txt`

**Command:**
```bash
pip install inference-sdk
```

---

### Task 3.2: Create Roboflow Building Detector
**Status:** Not Started  
**Priority:** CRITICAL  
**Estimated Time:** 1 hour

**Description:** Create new `RoboflowBuildingDetector` class that wraps Roboflow API.

**Deliverables:**
- [ ] Create `models/roboflow_building_detector.py`
- [ ] Load API key from `ROBOFLOW_API_KEY` env var
- [ ] Load workspace/project/version from env vars
- [ ] Implement `predict()` method returning `ToolOutput`
- [ ] Handle API errors explicitly (no silent failures)
- [ ] Apply confidence threshold from `BUILDING_CONFIDENCE_THRESHOLD`
- [ ] Filter by minimum area from `BUILDING_MIN_AREA`
- [ ] Log all API calls and responses

**Files to Create:**
- `models/roboflow_building_detector.py`

**Example Structure:**
```python
import os
from inference_sdk import InferenceHTTPClient
from models.tool_output_schema import ToolOutput, Detection
from utils.logger import setup_logger

logger = setup_logger(__name__)

class RoboflowBuildingDetector:
    def __init__(self):
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY not set in .env")
        
        self.workspace = os.getenv("ROBOFLOW_WORKSPACE")
        self.project = os.getenv("ROBOFLOW_PROJECT")
        self.version = os.getenv("ROBOFLOW_VERSION", "1")
        
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=self.api_key
        )
        
        self.confidence_threshold = float(os.getenv("BUILDING_CONFIDENCE_THRESHOLD", "0.4"))
        self.min_area = int(os.getenv("BUILDING_MIN_AREA", "100"))
        
        logger.info(f"Roboflow Building Detector initialized: {self.workspace}/{self.project}/{self.version}")
    
    async def predict(self, image, query, parameters):
        # Implement Roboflow API call
        # Return ToolOutput
        pass
```

---

### Task 3.3: Integrate Roboflow Detector into Execution Engine
**Status:** Not Started  
**Priority:** HIGH  
**Estimated Time:** 15 minutes

**Description:** Register Roboflow detector in execution engine's model registry.

**Acceptance Criteria:**
- [ ] Add `roboflow_building_detector` to model registry
- [ ] Update tool_selector to route `building_detection` to Roboflow
- [ ] Fallback to U-Net detector if Roboflow fails

**Files to Modify:**
- `agentic_layer/execution_engine.py`
- `agentic_layer/tool_selector.py`

---

### Task 3.4: Test Roboflow Building Detection
**Status:** Not Started  
**Priority:** CRITICAL  
**Estimated Time:** 30 minutes

**Description:** Test building detection on reference image with 18 buildings.

**Acceptance Criteria:**
- [ ] Detect 15+ buildings in reference image
- [ ] Each detection has confidence > 0.4
- [ ] Bounding boxes are accurate (visual inspection)
- [ ] Response time < 3 seconds
- [ ] Status = "ok" when buildings found
- [ ] Status = "no_detections" when none found (not "failed")

**Test Command:**
```bash
# Test via API
curl -X POST http://localhost:8000/api/query \
  -F "image=@data/raw/test_satellite_image.png" \
  -F "query=count buildings in this image"
```

**User Action Required:**
- [ ] Provide Roboflow workspace name
- [ ] Provide Roboflow project name
- [ ] Confirm model version
- [ ] Provide reference test image path
- [ ] Validate detection accuracy visually

---

## Phase 4: Spectral Index Model (Water & Vegetation)

### Task 4.1: Create Spectral Index Model
**Status:** Not Started  
**Priority:** HIGH  
**Estimated Time:** 1.5 hours

**Description:** Create `SpectralIndexModel` for NDVI and NDWI calculations.

**Deliverables:**
- [ ] Create `models/spectral_index_model.py`
- [ ] Implement NDVI: `(NIR - Red) / (NIR + Red)`
- [ ] Implement NDWI: `(Green - NIR) / (Green + NIR)`
- [ ] Load band indices from env vars (`NIR_BAND_INDEX`, etc.)
- [ ] Load thresholds from env vars (`NDVI_THRESHOLD`, `NDWI_THRESHOLD`)
- [ ] RGB fallback: check if NIR band exists
  - If not, return `status = "not_applicable"` with warning
  - Never fabricate results from RGB-only data
- [ ] Convert binary mask to bounding boxes using OpenCV contours
- [ ] Return `ToolOutput` with `type = "segmentation"`

**Files to Create:**
- `models/spectral_index_model.py`

**Example Structure:**
```python
import numpy as np
import cv2
from models.tool_output_schema import ToolOutput, Detection
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SpectralIndexModel:
    def __init__(self):
        self.nir_band = int(os.getenv("NIR_BAND_INDEX", "3"))
        self.red_band = int(os.getenv("RED_BAND_INDEX", "0"))
        self.green_band = int(os.getenv("GREEN_BAND_INDEX", "1"))
        self.ndvi_threshold = float(os.getenv("NDVI_THRESHOLD", "0.3"))
        self.ndwi_threshold = float(os.getenv("NDWI_THRESHOLD", "0.2"))
    
    async def predict(self, image, query, parameters):
        index_type = parameters.get("index_type", "ndvi")
        
        # Check if multispectral (has NIR band)
        if image.shape[2] <= max(self.nir_band, self.red_band, self.green_band):
            logger.warning(f"{index_type.upper()} requires NIR band, not available in RGB image")
            return ToolOutput(
                type="segmentation",
                detections=[],
                mask=None,
                answer_text=None,
                status="not_applicable",
                model_name=f"Spectral Index Model ({index_type.upper()})",
                execution_time=0.0
            )
        
        if index_type == "ndvi":
            mask = self._compute_ndvi(image)
        elif index_type == "ndwi":
            mask = self._compute_ndwi(image)
        
        # Convert mask to bounding boxes
        detections = self._mask_to_detections(mask, index_type)
        
        return ToolOutput(
            type="segmentation",
            detections=detections,
            mask=mask,
            answer_text=None,
            status="ok" if detections else "no_detections",
            model_name=f"Spectral Index Model ({index_type.upper()})",
            execution_time=0.0
        )
    
    def _compute_ndvi(self, image):
        # Implement NDVI calculation
        pass
    
    def _compute_ndwi(self, image):
        # Implement NDWI calculation
        pass
    
    def _mask_to_detections(self, mask, label):
        # Use cv2.findContours to extract bounding boxes
        pass
```

---

### Task 4.2: Integrate Spectral Model into Execution Engine
**Status:** Not Started  
**Priority:** HIGH  
**Estimated Time:** 15 minutes

**Description:** Register spectral index model in execution engine.

**Acceptance Criteria:**
- [ ] Add `spectral_index_model` to model registry
- [ ] Tool selector routes water/vegetation intents to spectral model
- [ ] Pass `index_type` parameter correctly ("ndvi" or "ndwi")

**Files to Modify:**
- `agentic_layer/execution_engine.py`

---

### Task 4.3: Test NDVI (Vegetation Detection)
**Status:** Not Started  
**Priority:** HIGH  
**Estimated Time:** 30 minutes

**Description:** Test vegetation detection on multispectral image.

**Acceptance Criteria:**
- [ ] Query "analyze vegetation" routes to spectral_index_model
- [ ] NDVI mask generated correctly
- [ ] Bounding boxes extracted from mask
- [ ] RGB-only image returns `status = "not_applicable"`
- [ ] Multispectral image returns `status = "ok"` with detections

**Test Command:**
```bash
curl -X POST http://localhost:8000/api/query \
  -F "image=@data/raw/multispectral_image.tif" \
  -F "query=analyze vegetation in this image"
```

**User Action Required:**
- [ ] Provide multispectral test image with NIR band
- [ ] Provide RGB-only test image (for fallback testing)

---

### Task 4.4: Test NDWI (Water Detection)
**Status:** Not Started  
**Priority:** HIGH  
**Estimated Time:** 30 minutes

**Description:** Test water detection on multispectral image.

**Acceptance Criteria:**
- [ ] Query "locate water bodies" routes to spectral_index_model
- [ ] NDWI mask generated correctly
- [ ] Bounding boxes extracted from mask
- [ ] RGB-only image returns `status = "not_applicable"`
- [ ] Multispectral image returns `status = "ok"` with detections

**Test Command:**
```bash
curl -X POST http://localhost:8000/api/query \
  -F "image=@data/raw/multispectral_image.tif" \
  -F "query=locate water bodies"
```

---

## Phase 5: Result Aggregator & Evidence Compiler

### Task 5.1: Update Result Aggregator for Standardized Schema
**Status:** Not Started  
**Priority:** HIGH  
**Estimated Time:** 45 minutes

**Description:** Rewrite result aggregator to process standardized `ToolOutput`.

**Acceptance Criteria:**
- [ ] Extract text summaries from `ToolOutput` uniformly
- [ ] Build text-only prompt for Groq LLM
- [ ] Include: detection counts, confidence scores, status
- [ ] Never pass raw image data to LLM
- [ ] Handle all status types: ok, no_detections, not_applicable, failed
- [ ] LLM prompt makes clear it's synthesizing pre-computed results

**Files to Modify:**
- `agentic_layer/result_aggregator.py`

**Example Prompt Template:**
```
Vision Model Results:
- Model: {model_name}
- Status: {status}
- Detections: {count} {label}(s) found
- Top 3 Confidence Scores: {conf1}, {conf2}, {conf3}
- Average Confidence: {avg_conf}

User Query: "{query}"

Synthesize a natural language answer based on the vision model results above.
Do not claim you cannot see the image - you have the pre-computed analysis results.
```

---

### Task 5.2: Update Evidence Compiler for Standardized Schema
**Status:** Not Started  
**Priority:** MEDIUM  
**Estimated Time:** 30 minutes

**Description:** Update evidence compiler to generate visualizations from `ToolOutput`.

**Acceptance Criteria:**
- [ ] Extract detections from standardized schema
- [ ] Generate bounding box overlays from `detections` field
- [ ] Generate mask overlays from `mask` field (if present)
- [ ] Always include `bounding_boxes` in API response
- [ ] Set explicit status field in response

**Files to Modify:**
- `visualization/evidence_compiler.py`

---

### Task 5.3: Test End-to-End Integration
**Status:** Not Started  
**Priority:** CRITICAL  
**Estimated Time:** 1 hour

**Description:** Test complete pipeline with all detection types.

**Test Cases:**
- [ ] Building detection query → Roboflow → 15+ buildings detected
- [ ] Water detection query → NDWI → water bodies located
- [ ] Vegetation query → NDVI → vegetation areas identified
- [ ] General VQA query → BLIP → natural language answer
- [ ] RGB-only image with NDVI query → `not_applicable` status
- [ ] Empty image → `no_detections` status
- [ ] API error → `failed` status propagated correctly

**Acceptance Criteria:**
- [ ] All query types work end-to-end
- [ ] Correct tool routing logged
- [ ] Standardized responses returned
- [ ] Bounding boxes visualized in frontend
- [ ] No silent failures or fake data

---

## Phase 6: Documentation & Configuration

### Task 6.1: Update .env.example
**Status:** Not Started  
**Priority:** MEDIUM  
**Estimated Time:** 15 minutes

**Description:** Document all new environment variables.

**Acceptance Criteria:**
- [ ] Add all Roboflow variables with descriptions
- [ ] Add all detection threshold variables
- [ ] Add band index variables
- [ ] Include example values
- [ ] Add comments explaining each parameter

**Files to Modify:**
- `.env.example`

---

### Task 6.2: Update README
**Status:** Not Started  
**Priority:** LOW  
**Estimated Time:** 30 minutes

**Description:** Document the multi-tool architecture in README.

**Acceptance Criteria:**
- [ ] Describe intent-based routing
- [ ] List supported detection types
- [ ] Explain tool selection logic
- [ ] Document configuration options
- [ ] Add example queries for each detection type

**Files to Modify:**
- `README.md`

---

## Summary

**Total Estimated Time:** ~10 hours

**Critical Path:**
1. Validate existing intent routing (30 min)
2. Implement standardized schema (2 hours)
3. Integrate Roboflow building detection (2 hours)
4. Implement spectral index model (2 hours)
5. Update aggregator/compiler (1.5 hours)
6. End-to-end testing (1 hour)

**Current Blockers:**
- [ ] User needs to provide Roboflow workspace/project names
- [ ] User needs to provide test images (multispectral with NIR, reference building image)

**Next Immediate Action:**
Ask user for Roboflow configuration details and test image paths.
