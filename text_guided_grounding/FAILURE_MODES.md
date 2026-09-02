# Text-Guided Grounding Specialist Model — Failure Modes & Edge Case Analysis

**Component:** Specialist model layer — Text-guided grounding model (`text_guided_grounding/`)
**Target Datasets:** DIOR-RSVG, OPT-RSVG, VRSBench-VG

This document outlines known performance boundaries, out-of-distribution edge cases, and failure modes to inform the Tool Selector & Orchestration Engine.

---

## 1. Out-of-Distribution (OOD) Sensor Modalities (SAR & Thermal Imagery)

### Description
The text-guided grounding model is trained and fine-tuned exclusively on optical satellite imagery (DIOR-RSVG / OPT-RSVG RGB channels). When presented with Synthetic Aperture Radar (SAR / Sentinel-1) or Thermal Infrared (TIR) imagery:
- **Symptom**: Feature contrast differs radically from RGB visual characteristics, causing lower sigmoid logit activations.
- **Model Behavior**: The model yields low confidence scores (< 0.25) rather than producing hallucinated bounding boxes.
- **Orchestration Guidance**: The Tool Selector (`agentic_layer/tool_selector.py`) automatically routes SAR-containing queries to the `sar_fusion_model` primary tool under the **USP-1 Directive**.

---

## 2. Low-Confidence & Non-Existent Feature Queries

### Description
When a natural language query describes an object or feature that is not present in the satellite image frame (e.g., "a aircraft carrier in landlocked forest"):
- **Model Behavior**: Neural logits remain low across all candidate grid anchors.
- **Handling**: The model yields `confidence < 0.20` and returns candidate proposals with low score weights, correctly indicating absence of target features without hardcoded string-matching bypasses.

---

## 3. Sub-10px Small Target Objects

### Description
Grounding very small objects (< 10×10 pixels in high-resolution imagery, such as individual vehicles or small solar panels):
- **Symptom**: Feature representations in Patch32 vision transformer backbones undergo spatial downsampling, leading to coarse bounding box predictions.
- **Mitigation**: Pair text-guided grounding with multi-scale sliding window crops or specialized building/object detectors (`models/building_detector.py`).

---

## 4. Ambiguous & Relational Referring Expressions

### Description
Queries containing complex spatial relations (e.g., "the third building to the right of the water tank"):
- **Model Behavior**: Surfaces top-k candidate region proposals sorted by calibrated confidence scores.
- **Downstream Aggregation**: The `ResultAggregator` renders top-k candidate overlays for human analyst verification.
