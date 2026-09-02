# Multi-Tool Architecture Requirements

## Overview
Refactor SatQuery from a single generic grounding model architecture to an intent-based multi-tool system where different query types route to specialized detection modules optimized for their specific domain (buildings, water, vegetation).

## Problem Statement
Current implementation uses OWL-ViT zero-shot grounding for all detection tasks, resulting in poor accuracy:
- Building detection: 0-5 buildings detected vs. reference image showing 18 buildings
- Water/vegetation detection: Generic grounding not suitable for spectral analysis
- All query types forced through one generic model regardless of intent

**Root Cause:** OWL-ViT is trained on ground-level photos (COCO dataset), not satellite imagery. Zero-shot detection doesn't work for aerial views.

## Academic Context
- **Project**: VIT 3rd Semester ISIH
- **Deliverable Type**: Academic project with demo
- **Timeline Constraint**: Need working implementation incrementally, module by module
- **Dataset Constraints**: No time for training from scratch; must use pretrained models or APIs

## Functional Requirements

### FR-1: Intent Classification
**Priority:** HIGH  
**Status:** ✅ IMPLEMENTED

The system must classify user queries into specific intents:
- `building_detection` - Count/locate/identify buildings, structures, warehouses, factories
- `water_detection` - Locate water bodies, rivers, lakes, oceans
- `vegetation_detection` - Analyze vegetation, forests, crops, greenery, NDVI
- `change_detection` - Compare temporal changes, before/after analysis
- `general_vqa` - Generic visual questions not matching above categories

**Acceptance Criteria:**
- Query "count buildings in this image" → `building_detection`
- Query "locate water bodies" → `water_detection`  
- Query "what is in this image" → `general_vqa`
- Intent classification logged before tool routing

### FR-2: Intent-Based Tool Routing
**Priority:** HIGH  
**Status:** ✅ IMPLEMENTED (needs validation)

The system must route queries to appropriate specialist tools based on classified intent:

| Intent | Tool | Rationale |
|--------|------|-----------|
| building_detection | building_detector | Roboflow API specialized for satellite building detection |
| water_detection | spectral_index_model (NDWI) | Band-math approach for water body identification |
| vegetation_detection | spectral_index_model (NDVI) | Normalized Difference Vegetation Index |
| change_detection | change_detection_model | Temporal comparison |
| general_vqa | vqa_model | BLIP VQA for open-ended questions |

**Acceptance Criteria:**
- Tool selection respects LLM's intent classification (no hardcoded overrides)
- Logs show: "Intent routing: building_detection → building_detector"
- No fallback to grounding_model for specialized intents

### FR-3: Building Detection Accuracy
**Priority:** CRITICAL  
**Status:** ⚠️ NEEDS ROBOFLOW INTEGRATION

**Target Accuracy:** Detect 15-20 buildings in reference satellite images (comparable to manual annotation showing 18 buildings)

**Current State:**
- U-Net segmentation model exists (`building_detector.py`) but untested
- OWL-ViT zero-shot: 0-5 buildings (unacceptable for demo)

**Solution:** Integrate Roboflow Inference API
- **API Provider:** Roboflow
- **API Keys Available:** 
  - Private: `nFr9z8OUTQCmKOKgrt0c`
  - Publishable: `rf_tlTqe7gJV9NTmhgi42uaO7eSqn22`
- **Model Type:** Pretrained satellite building detection model from Roboflow Universe
- **Fallback:** If Roboflow fails, fall back to U-Net model (existing)

**Acceptance Criteria:**
- Detect 15+ buildings in reference test image (satellite image with 18 ground-truth buildings)
- Each detection includes: bounding box `[x1, y1, x2, y2]`, confidence score, label "building"
- Confidence threshold: 0.4 (configurable via `BUILDING_CONFIDENCE_THRESHOLD`)
- Response time: < 3 seconds per image
- Failed detections return explicit error status, never fake/dummy data

### FR-4: Water Detection via NDWI
**Priority:** HIGH  
**Status:** ⚠️ NOT IMPLEMENTED

**Method:** Normalized Difference Water Index (NDWI)
```
NDWI = (Green - NIR) / (Green + NIR)
```

**Requirements:**
- Input: Multispectral image with NIR and Green bands
- Output: Binary mask where NDWI > threshold (configurable, default 0.2)
- Mask-to-bbox conversion using OpenCV contours
- **RGB Fallback:** If NIR band unavailable, return status `not_applicable` with clear warning. Never fabricate results from RGB-only data.

**Acceptance Criteria:**
- Detect water bodies in multispectral test image
- Return standardized schema with detections/mask
- RGB-only input logs: "WARNING: NDWI requires NIR band, not available in RGB image" and returns status `not_applicable`

### FR-5: Vegetation Detection via NDVI
**Priority:** HIGH  
**Status:** ⚠️ NOT IMPLEMENTED

**Method:** Normalized Difference Vegetation Index (NDVI)
```
NDVI = (NIR - Red) / (NIR + Red)
```

**Requirements:**
- Input: Multispectral image with NIR and Red bands
- Output: Binary mask where NDVI > threshold (configurable, default 0.3)
- Mask-to-bbox conversion using OpenCV contours
- **RGB Fallback:** If NIR band unavailable, return status `not_applicable` with clear warning

**Acceptance Criteria:**
- Detect vegetation areas in multispectral test image
- Return standardized schema with detections/mask
- RGB-only input logs warning and returns `not_applicable`

### FR-6: Standardized Tool Output Schema
**Priority:** CRITICAL  
**Status:** ⚠️ NOT IMPLEMENTED

**Problem:** Each tool returns different output structure, making aggregation brittle and requiring special-casing.

**Solution:** All tools must return this standardized schema:

```python
{
    "type": "detection" | "segmentation" | "vqa",
    "detections": [
        {
            "label": str,           # e.g., "building", "water", "vegetation"
            "confidence": float,    # 0.0 - 1.0
            "box": [x1, y1, x2, y2] # pixel coordinates
        }
    ] or [],
    "mask": <np.ndarray or None>,   # Optional 2D binary mask for segmentation
    "answer_text": str or None,     # Optional natural language answer (VQA tools)
    "status": "ok" | "no_detections" | "not_applicable" | "failed",
    "model_name": str,              # e.g., "roboflow-building-detector"
    "execution_time": float         # seconds
}
```

**Acceptance Criteria:**
- `building_detector` returns schema with `type: "detection"`
- `spectral_index_model` returns schema with `type: "segmentation"` and `mask` field
- `vqa_model` returns schema with `type: "vqa"` and `answer_text` field
- `result_aggregator.py` processes all tools uniformly without tool-specific logic
- `evidence_compiler.py` generates visualizations from standardized schema

### FR-7: Result Aggregator Text-Only Prompting
**Priority:** HIGH  
**Status:** ✅ IMPLEMENTED (needs validation with new tools)

**Requirement:** Groq LLM is text-only. Result aggregator must:
1. Extract text summaries from standardized tool outputs
2. Build text-only prompt describing vision model results
3. Never pass raw image data to LLM
4. Never let LLM respond "I cannot see the image"

**Example Prompt:**
```
Vision Model Results:
- Building Detector (Roboflow): Detected 18 buildings with average confidence 0.87
  Top detections: building (0.92), building (0.89), building (0.85)
- Status: ok

User Query: "How many buildings are in this image?"

Synthesize a natural language answer based on the vision model results above.
```

**Acceptance Criteria:**
- All LLM prompts are text-only (no image data)
- Prompt explicitly includes detection counts, confidence scores, status
- LLM responses synthesize from provided data, never claim inability to see image

### FR-8: Always-Present Bounding Box Visualization
**Priority:** MEDIUM  
**Status:** ✅ IMPLEMENTED

**Requirement:** Frontend "Bounding Boxes" tab must always render, with explicit status:
- **ok:** Show bounding boxes overlay with detection count
- **no_detections:** Show "No detections found" message
- **not_applicable:** Show "Detection not applicable for this image type"
- **failed:** Show error message

API response must always include `bounding_boxes` field, never omit.

**Acceptance Criteria:**
- API response includes `bounding_boxes` even when empty
- Frontend tab shows appropriate message for each status
- Never hide tab or field from API response

## Non-Functional Requirements

### NFR-1: No Silent Failures
**Priority:** CRITICAL

**Requirement:** No tool shall substitute fake/dummy/simulated data on failure.

**Enforcement:**
- All failures raise typed exceptions (e.g., `ModelLoadError`, `InferenceError`)
- Errors logged as `ERROR` level with full traceback
- Standardized schema returns `status: "failed"` with error message
- Upstream code (orchestrator, aggregator) handles failures explicitly

**Acceptance Criteria:**
- Zero instances of hardcoded placeholder results
- All error paths tested and verified to propagate properly
- Logs show ERROR entries for all tool failures

### NFR-2: Configurable Parameters
**Priority:** MEDIUM

All detection thresholds and model parameters must be configurable via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ROBOFLOW_API_KEY` | (required) | Roboflow authentication |
| `ROBOFLOW_WORKSPACE` | your-workspace | Roboflow workspace name |
| `ROBOFLOW_PROJECT` | building-detection | Roboflow project name |
| `ROBOFLOW_VERSION` | 1 | Model version |
| `BUILDING_CONFIDENCE_THRESHOLD` | 0.4 | Building detection confidence cutoff |
| `NDVI_THRESHOLD` | 0.3 | Vegetation mask threshold |
| `NDWI_THRESHOLD` | 0.2 | Water mask threshold |
| `NIR_BAND_INDEX` | 3 | NIR band position (0-indexed) |
| `RED_BAND_INDEX` | 0 | Red band position |
| `GREEN_BAND_INDEX` | 1 | Green band position |
| `BUILDING_MIN_AREA` | 100 | Min pixels for valid building |

**Acceptance Criteria:**
- All parameters loaded from `.env` file
- Clear error messages if required keys missing
- Parameters documented in `.env.example`

### NFR-3: Incremental Testing
**Priority:** HIGH

**Requirement:** Implement and test one module at a time. After each module, pause for user validation with test images before proceeding.

**Module Sequence:**
1. ✅ Module 1: Intent classification & routing (DONE - needs validation)
2. Module 2: Standardized output schema (refactor existing tools)
3. Module 3: Roboflow building detector integration
4. Module 4: Spectral index model (NDVI/NDWI)
5. Module 5: Result aggregator update
6. Module 6: Evidence compiler update
7. Module 7: End-to-end integration test

**Acceptance Criteria:**
- User explicitly approves each module before next one starts
- Test images used: reference satellite image with 18 buildings, multispectral image with water/vegetation
- Each module has clear pass/fail criteria

### NFR-4: Performance
**Priority:** MEDIUM

- Building detection: < 3 seconds per image
- Spectral analysis: < 1 second per image
- Total pipeline (query → answer): < 5 seconds
- Memory usage: < 2GB RAM per inference

### NFR-5: Dependency Management
**Priority:** HIGH

**New Dependencies Required:**
- `inference-sdk` (Roboflow Python SDK)
- `opencv-python` (if not already present - for contour detection)
- No new heavy dependencies (no model retraining libraries)

**Action Items for User:**
- [ ] Provide Roboflow workspace name (update `ROBOFLOW_WORKSPACE` in `.env`)
- [ ] Provide Roboflow project name (update `ROBOFLOW_PROJECT` in `.env`)
- [ ] Confirm Roboflow model version (update `ROBOFLOW_VERSION` in `.env`)
- [ ] Provide test images: 
  - [ ] Reference satellite image with 18 buildings (for validation)
  - [ ] Multispectral image with NIR band (for NDVI/NDWI testing)
  - [ ] RGB-only satellite image (for RGB fallback testing)

## Out of Scope
- Training new models from scratch
- Fine-tuning existing models (no time/GPU resources)
- Real-time video processing
- 3D reconstruction
- Multi-temporal stack analysis (beyond simple before/after comparison)
- SAR fusion implementation (existing code kept as-is)

## Success Criteria
1. **Building Detection:** 15+ buildings detected in reference image (vs. current 0-5)
2. **Intent Routing:** Logs confirm intent → tool routing works correctly
3. **No Silent Failures:** All error paths tested and verified
4. **Standardized Schema:** All tools return uniform output structure
5. **Demo Ready:** Can demonstrate "count buildings", "locate water", "analyze vegetation" queries end-to-end

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Roboflow API quota limits | HIGH | Implement local U-Net fallback |
| Roboflow model not accurate enough | HIGH | Fine-tune U-Net on SpaceNet subset as backup plan |
| RGB-only images (no NIR) for NDVI/NDWI | MEDIUM | Explicit `not_applicable` status, never fake results |
| Tool output schema changes break frontend | MEDIUM | Version schema, test with existing frontend before full rollout |
| LLM hallucinating results despite text-only prompt | LOW | Add confidence scores to prompt, validate LLM output format |

## Questions for User
1. What is your Roboflow workspace name? (needed for API calls)
2. What is your Roboflow project name for building detection? (or should we browse Roboflow Universe for a public model?)
3. Do you have multispectral test images with NIR bands, or only RGB satellite images?
4. What is the reference test image path we should use for validation?
5. Is there a specific accuracy metric you need to report for the academic deliverable (e.g., mAP, precision, recall)?
