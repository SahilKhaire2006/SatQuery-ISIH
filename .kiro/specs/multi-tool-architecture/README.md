# Multi-Tool Architecture Spec

## Overview
This spec guides the refactoring of SatQuery from a single generic grounding model to an intent-based multi-tool architecture with specialized detection modules.

## Spec Structure

### 📋 requirements.md
Comprehensive requirements documentation including:
- Problem statement and academic context
- Functional requirements (FR-1 through FR-8)
- Non-functional requirements (NFR-1 through NFR-5)
- Success criteria and risk assessment
- Questions for user

**Status:** ✅ COMPLETE - Ready for user review

### 📊 spec.json
Structured metadata for tooling integration:
- Spec version and ownership
- Module tracking with status
- Dependency mapping
- Environment variable requirements

**Status:** ✅ COMPLETE

### ✅ tasks.md
Detailed implementation tasks organized by phase:
- Phase 1: Foundation & Validation (30 min)
- Phase 2: Standardized Output Schema (2 hours)
- Phase 3: Roboflow Building Detection (2 hours)
- Phase 4: Spectral Index Model (2 hours)
- Phase 5: Result Aggregator & Evidence Compiler (1.5 hours)
- Phase 6: Documentation (45 min)

**Total Estimated Time:** ~10 hours  
**Status:** ✅ COMPLETE - Ready for execution

## Current State

### ✅ Already Implemented
1. **Intent Classification** (`agentic_layer/query_interpreter.py`)
   - Classifies queries into: building_detection, water_detection, vegetation_detection, change_detection, general_vqa
   
2. **Intent-Based Routing** (`agentic_layer/tool_selector.py`)
   - Routes intents to appropriate tools
   - Needs validation that LLM's classification is respected

3. **Building Detector Skeleton** (`models/building_detector.py`)
   - U-Net segmentation model exists
   - Untested, needs Roboflow integration

4. **Bounding Box Visualization** (frontend + orchestrator)
   - Always renders, shows status messages

### ⚠️ Needs Implementation
1. **Standardized Tool Output Schema** - CRITICAL
2. **Roboflow API Integration** - CRITICAL for accuracy
3. **Spectral Index Model** (NDVI/NDWI) - HIGH priority
4. **Result Aggregator Update** - HIGH priority
5. **Evidence Compiler Update** - MEDIUM priority

## Configuration Added

New environment variables added to `.env`:

```env
# Roboflow Configuration
ROBOFLOW_API_KEY=nFr9z8OUTQCmKOKgrt0c
ROBOFLOW_PUBLISHABLE_KEY=rf_tlTqe7gJV9NTmhgi42uaO7eSqn22
ROBOFLOW_WORKSPACE=your-workspace  # ⚠️ User needs to provide
ROBOFLOW_PROJECT=building-detection  # ⚠️ User needs to provide
ROBOFLOW_VERSION=1

# Detection Configuration
BUILDING_CONFIDENCE_THRESHOLD=0.4
NDVI_THRESHOLD=0.3
NDWI_THRESHOLD=0.2
BUILDING_MIN_AREA=100
```

## User Actions Required

Before starting implementation, user must provide:

1. **Roboflow Configuration**
   - [ ] Workspace name (or choose public model from Roboflow Universe)
   - [ ] Project name for building detection
   - [ ] Confirm model version (default: 1)

2. **Test Images**
   - [ ] Reference satellite image with 18 buildings (for validation)
   - [ ] Multispectral image with NIR band (for NDVI/NDWI testing)
   - [ ] RGB-only satellite image (for fallback testing)
   - [ ] Provide file paths for these images

3. **Accuracy Requirements**
   - [ ] Confirm target: 15+ building detections (vs. current 0-5)
   - [ ] Any specific metrics needed for academic deliverable? (mAP, precision, recall)

## Next Steps

### Immediate Actions
1. **User Review:** Review `requirements.md` and confirm all requirements are correct
2. **User Input:** Provide Roboflow configuration and test image paths
3. **Start Implementation:** Begin with Task 1.1 (Validate Intent Classification)

### Implementation Strategy
- **Incremental:** Complete one phase at a time
- **Test After Each Module:** User validates before proceeding to next
- **No Silent Failures:** All errors explicit and logged
- **Fallback Paths:** Roboflow → U-Net, Multispectral → RGB warning

### Success Metrics
- ✅ Building detection: 15+ buildings (vs. 0-5 currently)
- ✅ Intent routing works correctly (logs confirm)
- ✅ Water/vegetation detection functional (or explicit "not_applicable")
- ✅ No fake/dummy data on failures
- ✅ Demo-ready for academic presentation

## Questions?

If anything in the requirements or tasks is unclear, ask before starting implementation. The spec is a living document - update it as we learn more during implementation.

## Spec Version
- **Version:** 1.0.0
- **Created:** 2026-09-02
- **Status:** Requirements Phase - Ready for User Review
- **Next Phase:** Design (will be created after user confirms requirements)
