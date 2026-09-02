# Quick Start Guide - Multi-Tool Architecture

## What We're Building

Transform SatQuery from generic object detection (0-5 buildings detected) to specialized detection modules (target: 15+ buildings).

## Current Status

✅ **DONE:**
- Roboflow API keys added to `.env`
- Intent classification implemented
- Intent-based routing implemented
- Spec files created with detailed requirements and tasks

⚠️ **NEEDS YOUR INPUT:**
1. Roboflow workspace name (update `ROBOFLOW_WORKSPACE` in `.env`)
2. Roboflow project name (update `ROBOFLOW_PROJECT` in `.env`)
3. Test image paths (where is the reference image with 18 buildings?)

## What You Need to Provide

### 1. Roboflow Configuration

**Option A: Use Your Own Roboflow Model**
- Go to https://app.roboflow.com
- Find your workspace name (in URL or dashboard)
- Find your building detection project name
- Update `.env` with these values

**Option B: Use Public Roboflow Model**
- Browse https://universe.roboflow.com
- Search for "satellite building detection" or "aerial building detection"
- Note the workspace/project/version from model page
- Update `.env` with these values

### 2. Test Images

We need 3 test images:

**Image 1: Reference Building Image (CRITICAL)**
- The satellite image where you manually counted 18 buildings
- This is our validation target (need to detect 15+)
- **File path:** `?????` ← Please provide

**Image 2: Multispectral Image (for NDVI/NDWI)**
- Satellite image with NIR band (at least 4 bands: R, G, B, NIR)
- Format: `.tif` or multi-channel `.png`
- **File path:** `?????` ← Please provide (or tell me if you only have RGB)

**Image 3: RGB-Only Image (for fallback testing)**
- Regular satellite image (3 channels: RGB only)
- Used to test "not_applicable" status when NIR unavailable
- **File path:** `?????` ← Please provide

## Quick Config Check

### Check Your .env File

Open `.env` and verify:

```env
# Roboflow - API keys already added ✅
ROBOFLOW_API_KEY=nFr9z8OUTQCmKOKgrt0c
ROBOFLOW_PUBLISHABLE_KEY=rf_tlTqe7gJV9NTmhgi42uaO7eSqn22

# Roboflow - YOU NEED TO UPDATE THESE ⚠️
ROBOFLOW_WORKSPACE=your-workspace  # ← Change this
ROBOFLOW_PROJECT=building-detection  # ← Change this
ROBOFLOW_VERSION=1  # ← Confirm or change

# Detection thresholds - defaults are reasonable
BUILDING_CONFIDENCE_THRESHOLD=0.4  # Lower = more detections
NDVI_THRESHOLD=0.3
NDWI_THRESHOLD=0.2
BUILDING_MIN_AREA=100  # Minimum pixels for valid building
```

## Implementation Phases (10 hours total)

### Phase 1: Validate Existing Code (30 min)
- Test intent classification
- Test tool routing
- Confirm logs show correct intent → tool mapping

### Phase 2: Standardize Output Schema (2 hours)
- Create `ToolOutput` dataclass
- Refactor all models to return standardized output
- No more custom result formats per tool

### Phase 3: Roboflow Building Detection (2 hours) ⭐ CRITICAL
- Install `inference-sdk`
- Create `RoboflowBuildingDetector` class
- Test on reference image → target 15+ buildings

### Phase 4: Spectral Index Model (2 hours)
- Create `SpectralIndexModel` class
- Implement NDVI (vegetation) and NDWI (water)
- Handle RGB-only images gracefully

### Phase 5: Update Aggregator/Compiler (1.5 hours)
- Update result aggregator for standardized schema
- Update evidence compiler for new tool outputs
- Test end-to-end pipeline

### Phase 6: Documentation (45 min)
- Update `.env.example`
- Update README with new architecture

## Testing Strategy

After each phase, we test with real queries:

```bash
# Test building detection
curl -X POST http://localhost:8000/api/query \
  -F "image=@path/to/reference_image.png" \
  -F "query=count buildings in this image"

# Test water detection (needs multispectral image)
curl -X POST http://localhost:8000/api/query \
  -F "image=@path/to/multispectral.tif" \
  -F "query=locate water bodies"

# Test vegetation detection
curl -X POST http://localhost:8000/api/query \
  -F "image=@path/to/multispectral.tif" \
  -F "query=analyze vegetation"
```

## Key Principles

1. **No Silent Failures:** All errors logged and propagated
2. **Incremental Testing:** Validate each module before next
3. **Explicit Status:** Every result has status: ok/no_detections/not_applicable/failed
4. **No Fake Data:** Never substitute dummy results on failure

## What Happens Next?

Once you provide:
1. ✅ Roboflow workspace/project names
2. ✅ Test image paths

I will:
1. Start with Phase 1 (validate existing code)
2. Pause after each phase for your validation
3. Show you logs/results at each step
4. Proceed to next phase only after your approval

## Questions to Answer

Please respond with:

1. **Roboflow Config:**
   - Workspace: `?????`
   - Project: `?????`
   - Version: `?????` (or keep default `1`)

2. **Test Image Paths:**
   - Reference building image (18 buildings): `?????`
   - Multispectral image with NIR: `?????` (or "I only have RGB images")
   - RGB-only image: `?????`

3. **Any Accuracy Metrics Required?**
   - Do you need specific metrics for academic deliverable? (precision, recall, mAP, F1)
   - Or is visual validation + detection count sufficient?

## Ready to Start?

Once you provide the above info, I'll:
1. Update `.env` with your Roboflow config
2. Start Phase 1: Validate intent classification and routing
3. Show you logs confirming correct behavior
4. Wait for your "continue" before Phase 2

Let me know when you're ready! 🚀
