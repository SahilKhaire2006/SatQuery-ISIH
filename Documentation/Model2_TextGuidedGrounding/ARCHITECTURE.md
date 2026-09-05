# Model 2 — System Architecture & Data Flow

## High-Level Architecture

```mermaid
graph TD
    A["User Query (Text Only)"] --> B["API Gateway<br>/api/v1/query"]
    B --> C["Orchestrator"]
    C --> D["Query Interpreter<br>(Groq LLM)"]
    
    D -->|"intent: disaster_flood"| E["Tool Selector"]
    D -->|"intent: disaster_earthquake"| E
    D -->|"intent: disaster_general"| E
    
    E --> F["Disaster Grounding Model"]
    
    F --> G["Location Resolver"]
    G -->|"lat, lon, bbox"| H["Imagery Router"]
    
    H -->|"Primary"| I["Sentinel Hub API<br>(Sentinel-2 RGB + Sentinel-1 SAR)"]
    H -->|"Fallback"| J["Esri World Imagery<br>(existing map_fetcher.py)"]
    H -->|"Optional"| K["NASA Earthdata<br>(FIRMS, OPERA DSWx)"]
    
    I --> L["Real Satellite Image(s)"]
    J --> L
    K --> L
    
    L --> M["Disaster Analysis Pipeline"]
    
    M --> N["Flood Analyzer"]
    M --> O["Earthquake Analyzer"]
    M --> P["Scene Description<br>(Reuse Model 1 VQA)"]
    
    N --> Q["NDWI Water Mapping"]
    N --> R["Flood Progression<br>(Multi-Temporal)"]
    N --> S["Infrastructure Impact"]
    N --> T["Evacuation Zones"]
    
    O --> U["Change Detection<br>(Pre/Post)"]
    O --> V["Damage Classification"]
    O --> W["Landslide Risk"]
    
    Q --> X["LLM Disaster Reasoner<br>(Groq)"]
    R --> X
    S --> X
    T --> X
    U --> X
    V --> X
    W --> X
    P --> X
    
    X --> Y["Disaster Report + Visual Evidence"]
    Y --> Z["API Response"]
    Z --> AA["Frontend<br>DisasterAnalysisCard.jsx"]
```

---

## Data Flow — Flood Query Example

```
Input:  "Show flood progression in Wayanad, Kerala"

Step 1: Query Interpreter
  ├── intent: "disaster_flood"
  ├── location: "Wayanad, Kerala"
  ├── disaster_type: "flood"
  └── temporal: "progression" → needs multi-temporal

Step 2: Location Resolution
  ├── Geocode: "Wayanad, Kerala" → (11.6854, 76.1320)
  ├── BBox: [76.0, 11.5, 76.3, 11.9]
  └── Area: 5000m × 5000m (wider for disaster)

Step 3: Imagery Fetching
  ├── Sentinel-2 T-14d: RGB image (pre-flood)
  ├── Sentinel-2 T-7d:  RGB image (early flood)
  ├── Sentinel-2 T+0:   RGB image (current)
  ├── Sentinel-2 NDWI:  Water index for each date
  └── Sentinel-1 SAR:   Cloud-penetrating flood map

Step 4: Flood Analysis
  ├── NDWI thresholding → water mask at each timestep
  ├── Difference: T+0 minus T-14d → flood expansion map
  ├── Progression rate: ΔA/Δt = km²/day
  ├── Building detection within flood mask → affected count
  └── Safe zone identification → evacuation areas

Step 5: LLM Synthesis
  ├── Situation assessment
  ├── Escalation recommendation
  ├── Evacuation priorities
  ├── 24-48h progression prediction
  └── Resource deployment suggestions

Step 6: Response Assembly
  ├── Disaster report (text)
  ├── Flood extent overlay (visual evidence)
  ├── Temporal progression animation/carousel
  ├── Bounding boxes (affected infrastructure)
  ├── Evacuation zone map
  └── Full audit trail
```

---

## Integration Points with Existing System

```
EXISTING SYSTEM                          MODEL 2 ADDITIONS
─────────────────                        ──────────────────

┌─────────────────────┐
│   query_interpreter │
│   ├── building_det  │
│   ├── water_det     │                  ┌──────────────────────┐
│   ├── vegetation    │     + ADD ──────►│ disaster_flood       │
│   ├── change_det    │                  │ disaster_earthquake  │
│   └── general_vqa   │                  │ disaster_general     │
└─────────────────────┘                  └──────────────────────┘

┌─────────────────────┐
│   tool_selector     │
│   ├── vqa_model     │
│   ├── grounding     │
│   ├── roboflow_bd   │                  ┌──────────────────────────┐
│   ├── spectral_idx  │     + ADD ──────►│ disaster_grounding_model │
│   ├── change_det    │                  └──────────────────────────┘
│   └── sar_fusion    │
└─────────────────────┘

┌─────────────────────┐
│   execution_engine  │
│   ├── VQAModel      │
│   ├── GroundingModel│
│   ├── Roboflow      │                  ┌──────────────────────────┐
│   ├── SpectralIndex │     + ADD ──────►│ DisasterGroundingModel   │
│   ├── ChangeDet     │                  │ (wraps FloodAnalyzer,    │
│   └── SARFusion     │                  │  EarthquakeAnalyzer,     │
└─────────────────────┘                  │  DisasterLLMReasoner)    │
                                         └──────────────────────────┘

┌─────────────────────┐
│   geospatial/       │
│   ├── map_fetcher   │                  ┌──────────────────────────┐
│   ├── metadata_par  │     + ADD ──────►│ sentinel_fetcher.py      │
│   ├── coord_system  │                  │ imagery_router.py        │
│   └── tile_proc     │                  └──────────────────────────┘
└─────────────────────┘
```

---

## Satellite Imagery Fetch Decision Tree

```mermaid
graph TD
    A["Query with Location"] --> B{"Disaster Type?"}
    
    B -->|"Flood (active)"| C{"Weather Conditions?"}
    B -->|"Earthquake"| D["Fetch Pre+Post Optical"]
    B -->|"General"| E["Fetch Latest Optical"]
    
    C -->|"Cloudy/Raining"| F["Sentinel-1 SAR<br>(penetrates clouds)"]
    C -->|"Clear"| G["Sentinel-2 Optical<br>(higher resolution)"]
    
    D --> H{"Sentinel Hub Available?"}
    E --> H
    F --> H
    G --> H
    
    H -->|"Yes"| I["Use Sentinel Hub<br>Process API"]
    H -->|"No"| J["Fallback: Esri<br>World Imagery"]
    
    I --> K["Return Real Image<br>+ Metadata"]
    J --> K
```

---

## Module Dependency Graph

```
disaster_grounding_model.py
    ├── geospatial/sentinel_fetcher.py
    │     └── sentinelhub (pip package)
    ├── geospatial/imagery_router.py
    │     ├── sentinel_fetcher.py
    │     └── map_fetcher.py (existing)
    ├── models/disaster_analysis/flood_analyzer.py
    │     ├── models/spectral_index_model.py (existing, NDWI)
    │     └── models/roboflow_building_detector.py (existing)
    ├── models/disaster_analysis/earthquake_analyzer.py
    │     └── models/change_detection_model.py (existing)
    ├── models/disaster_analysis/disaster_llm_reasoner.py
    │     └── models/llm/groq_engine.py (existing)
    └── models/vqa_model.py (existing, for scene description)
```
