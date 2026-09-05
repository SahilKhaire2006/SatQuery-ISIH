# Model 2 — Text-Guided Grounding for Disaster Management

> **Purpose**: This directory contains all design documentation, architecture decisions, and action plans for **Model 2** of the SatQuery-ISIH system — a *text-only query model* that fetches **real-time satellite imagery** from Earth-observation APIs, and performs **disaster-specific analysis** (flood progression, earthquake damage, evacuation planning, predictions) alongside general scene description.

## Key Differentiator from Model 1

| Aspect | Model 1 (VQA + Captioning) | Model 2 (Text-Guided Grounding) |
|---|---|---|
| **Input** | User query + uploaded satellite image | User query only (no image) |
| **Image Source** | User-provided | Fetched automatically from real satellite APIs |
| **Scope** | Building / Water / Vegetation detection | Disaster management (Flood, Earthquake, etc.) |
| **Output** | VQA answer, bounding boxes, spectral indices | Disaster analysis, flood progression, evacuation plans, predictions + scene description |

## Documents in this Directory

| File | Description |
|---|---|
| [ACTION_PLAN.md](./ACTION_PLAN.md) | Complete phased action plan for development + integration |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System architecture and data flow diagrams |
| [SATELLITE_IMAGERY_SOURCES.md](./SATELLITE_IMAGERY_SOURCES.md) | Real-time satellite imagery API evaluation and selection |
| [DISASTER_ANALYSIS_PIPELINE.md](./DISASTER_ANALYSIS_PIPELINE.md) | Disaster-specific analysis pipeline (flood, earthquake) |
| [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) | How Model 2 integrates into the existing SatQuery system |
