# Disaster Analysis Pipeline — Technical Design

## Overview

The disaster analysis pipeline takes **real satellite imagery** (fetched by the imagery engine) and produces **actionable disaster intelligence** through a combination of spectral analysis, computer vision, and LLM-powered reasoning.

---

## Pipeline Architecture

```
Fetched Satellite Imagery
    │
    ├── Sentinel-2 RGB (optical)
    ├── Sentinel-2 NDWI (water index)
    ├── Sentinel-1 SAR (radar)
    └── Multi-temporal series
    │
    ▼
┌─────────────────────────────────────┐
│        DISASTER TYPE ROUTER         │
│   Based on intent from QueryInterp  │
└──────────┬──────────┬───────────────┘
           │          │
    ┌──────▼──┐  ┌────▼─────────┐
    │  FLOOD  │  │  EARTHQUAKE  │
    └──┬──────┘  └──┬───────────┘
       │            │
       ▼            ▼
┌─────────────┐ ┌────────────────┐
│ Flood       │ │ Earthquake     │
│ Analyzer    │ │ Analyzer       │
│             │ │                │
│ • NDWI map  │ │ • Pre/post     │
│ • SAR flood │ │   change det.  │
│ • Progress  │ │ • Damage class │
│ • Infra.    │ │ • Landslide    │
│ • Evac.     │ │   risk         │
└──────┬──────┘ └───────┬────────┘
       │                │
       └───────┬────────┘
               │
               ▼
┌──────────────────────────────┐
│  Scene Description           │
│  (Reuse Model 1 VQA pipeline)│
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  LLM Disaster Reasoner      │
│  (Groq Llama 3.3-70B)       │
│                              │
│  • Situation assessment      │
│  • Escalation level          │
│  • Evacuation priorities     │
│  • Progression prediction    │
│  • Resource recommendations  │
└──────────┬───────────────────┘
           │
           ▼
    Disaster Report
```

---

## 1. Flood Analysis Pipeline

### 1.1 Water Extent Detection

#### Method A: NDWI Thresholding (Sentinel-2)

```python
def compute_water_mask(ndwi_map: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    NDWI (Normalized Difference Water Index):
    NDWI = (Green - NIR) / (Green + NIR)
    
    Values > threshold are classified as water.
    Sentinel Hub computes NDWI server-side; we just threshold.
    """
    water_mask = (ndwi_map > threshold).astype(np.uint8)
    
    # Morphological cleanup: remove noise, fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
    
    return water_mask
```

#### Method B: SAR Backscatter Thresholding (Sentinel-1)

```python
def compute_sar_flood_mask(sar_vv: np.ndarray, threshold_db: float = -15) -> np.ndarray:
    """
    SAR flood detection:
    - Open water has very low backscatter (specular reflection)
    - VV polarization < threshold_db (typically -15 to -20 dB) → flood
    
    Advantage: Works through clouds, rain, and at night.
    """
    # Convert to dB if in linear scale
    sar_db = 10 * np.log10(np.clip(sar_vv, 1e-10, None))
    
    flood_mask = (sar_db < threshold_db).astype(np.uint8)
    
    # Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    flood_mask = cv2.morphologyEx(flood_mask, cv2.MORPH_CLOSE, kernel)
    
    return flood_mask
```

#### Method C: Fused Detection (NDWI + SAR)

```python
def fuse_flood_detection(ndwi_mask: np.ndarray, sar_mask: np.ndarray) -> np.ndarray:
    """
    Combine optical NDWI + SAR for robust flood mapping.
    A pixel is classified as flooded if EITHER source detects water.
    
    This handles:
    - Cloudy areas (NDWI unreliable → SAR provides data)
    - Shallow water (SAR unreliable → NDWI provides data)
    """
    fused = np.maximum(ndwi_mask, sar_mask)
    return fused
```

### 1.2 Flood Progression Analysis

```python
def analyze_flood_progression(
    temporal_ndwi_maps: List[Tuple[str, np.ndarray]],
    pixel_area_m2: float = 100  # 10m × 10m for Sentinel-2
) -> Dict:
    """
    Track flood expansion over time.
    
    Input: List of (date_str, ndwi_map) tuples ordered chronologically
    Output: Progression metrics + difference maps
    """
    progression = []
    
    for i in range(1, len(temporal_ndwi_maps)):
        date_prev, mask_prev = temporal_ndwi_maps[i-1][0], compute_water_mask(temporal_ndwi_maps[i-1][1])
        date_curr, mask_curr = temporal_ndwi_maps[i][0], compute_water_mask(temporal_ndwi_maps[i][1])
        
        # New flood areas (water in current but not in previous)
        new_flood = np.logical_and(mask_curr, ~mask_prev.astype(bool)).astype(np.uint8)
        
        # Receded areas (water in previous but not in current)
        receded = np.logical_and(mask_prev, ~mask_curr.astype(bool)).astype(np.uint8)
        
        # Compute areas
        new_flood_area_km2 = np.sum(new_flood) * pixel_area_m2 / 1e6
        receded_area_km2 = np.sum(receded) * pixel_area_m2 / 1e6
        total_flood_km2 = np.sum(mask_curr) * pixel_area_m2 / 1e6
        
        progression.append({
            'from_date': date_prev,
            'to_date': date_curr,
            'new_flood_area_km2': float(new_flood_area_km2),
            'receded_area_km2': float(receded_area_km2),
            'total_flood_km2': float(total_flood_km2),
            'net_change_km2': float(new_flood_area_km2 - receded_area_km2),
            'expansion_map': new_flood,
            'recession_map': receded
        })
    
    # Compute overall progression rate
    if len(progression) >= 2:
        total_expansion = sum(p['net_change_km2'] for p in progression)
        total_days = len(progression)  # Approximate
        rate_km2_per_day = total_expansion / max(total_days, 1)
    else:
        rate_km2_per_day = 0.0
    
    return {
        'progression': progression,
        'rate_km2_per_day': rate_km2_per_day,
        'trend': 'expanding' if rate_km2_per_day > 0.5 else ('receding' if rate_km2_per_day < -0.5 else 'stable')
    }
```

### 1.3 Infrastructure Impact Assessment

```python
def assess_infrastructure_impact(
    flood_mask: np.ndarray,
    rgb_image: np.ndarray,
    building_detector  # Existing Roboflow/OWL-ViT detector
) -> Dict:
    """
    Detect buildings and infrastructure within flood zone.
    Reuses existing building detection from Model 1.
    """
    # Run existing building detector on the RGB image
    detections = building_detector.predict(
        image=rgb_image, 
        query="buildings and structures", 
        parameters={'target_object': 'building'}
    )
    
    buildings = detections['output'].get('detections', [])
    
    # Check which buildings intersect with flood mask
    affected = []
    safe = []
    
    for bldg in buildings:
        bbox = bldg.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        
        # Check if building bbox overlaps with flood mask
        building_region = flood_mask[y1:y2, x1:x2]
        flood_overlap = np.mean(building_region) if building_region.size > 0 else 0
        
        if flood_overlap > 0.3:  # >30% of building area is flooded
            bldg['flood_status'] = 'affected'
            bldg['flood_overlap_pct'] = float(flood_overlap * 100)
            affected.append(bldg)
        else:
            bldg['flood_status'] = 'safe'
            safe.append(bldg)
    
    return {
        'total_structures': len(buildings),
        'affected_structures': len(affected),
        'safe_structures': len(safe),
        'affected_details': affected,
        'impact_severity': (
            'critical' if len(affected) > 50 else
            'high' if len(affected) > 20 else
            'moderate' if len(affected) > 5 else
            'low'
        )
    }
```

### 1.4 Evacuation Zone Estimation

```python
def estimate_evacuation_zones(
    flood_mask: np.ndarray,
    pixel_size_m: float = 10.0,
    buffer_m: float = 500.0
) -> Dict:
    """
    Identify safe zones and evacuation corridors.
    """
    # Compute distance from flood boundary
    dist_from_flood = cv2.distanceTransform(
        (1 - flood_mask).astype(np.uint8), 
        cv2.DIST_L2, 5
    ) * pixel_size_m  # Convert to meters
    
    # Zone classification
    danger_zone = (dist_from_flood < 100).astype(np.uint8)     # < 100m from flood
    warning_zone = ((dist_from_flood >= 100) & (dist_from_flood < 500)).astype(np.uint8)
    safe_zone = (dist_from_flood >= 500).astype(np.uint8)
    
    # Find largest contiguous safe area (potential evacuation center)
    contours, _ = cv2.findContours(safe_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    evacuation_centers = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            area_km2 = cv2.contourArea(cnt) * (pixel_size_m ** 2) / 1e6
            evacuation_centers.append({
                'center_pixel': [cx, cy],
                'area_km2': float(area_km2),
                'distance_from_flood_m': float(dist_from_flood[cy, cx])
            })
    
    return {
        'danger_zone_mask': danger_zone,
        'warning_zone_mask': warning_zone,
        'safe_zone_mask': safe_zone,
        'evacuation_centers': evacuation_centers,
        'safe_area_percentage': float(np.mean(safe_zone) * 100)
    }
```

---

## 2. Earthquake Analysis Pipeline

### 2.1 Structural Damage Detection

```python
def detect_structural_damage(
    pre_image: np.ndarray,
    post_image: np.ndarray
) -> Dict:
    """
    Compare pre and post earthquake imagery to detect structural changes.
    Uses pixel-level change detection + building detector intersection.
    """
    # Ensure same size
    if pre_image.shape != post_image.shape:
        post_image = cv2.resize(post_image, (pre_image.shape[1], pre_image.shape[0]))
    
    # Convert to grayscale for change detection
    pre_gray = cv2.cvtColor(pre_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    post_gray = cv2.cvtColor(post_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Compute absolute difference
    diff = np.abs(post_gray - pre_gray)
    
    # Normalize and threshold
    diff_normalized = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
    _, change_mask = cv2.threshold(diff_normalized, 50, 255, cv2.THRESH_BINARY)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)
    
    # Compute change statistics
    change_percentage = float(np.mean(change_mask > 0) * 100)
    
    return {
        'change_mask': change_mask,
        'change_intensity_map': diff_normalized,
        'change_percentage': change_percentage,
        'severity': (
            'catastrophic' if change_percentage > 30 else
            'severe' if change_percentage > 15 else
            'moderate' if change_percentage > 5 else
            'minor'
        )
    }
```

### 2.2 Damage Severity Zoning

```python
def classify_damage_zones(
    change_intensity: np.ndarray
) -> Dict:
    """
    Classify damage into severity zones based on change intensity.
    """
    zones = np.zeros_like(change_intensity, dtype=np.uint8)
    
    # Zone classification based on change intensity
    zones[change_intensity < 30] = 0    # No significant damage
    zones[(change_intensity >= 30) & (change_intensity < 80)] = 1   # Minor damage
    zones[(change_intensity >= 80) & (change_intensity < 150)] = 2  # Moderate damage
    zones[(change_intensity >= 150) & (change_intensity < 200)] = 3 # Severe damage
    zones[change_intensity >= 200] = 4  # Destroyed/collapsed
    
    zone_areas = {
        'no_damage': float(np.mean(zones == 0) * 100),
        'minor': float(np.mean(zones == 1) * 100),
        'moderate': float(np.mean(zones == 2) * 100),
        'severe': float(np.mean(zones == 3) * 100),
        'destroyed': float(np.mean(zones == 4) * 100),
    }
    
    return {
        'damage_zone_map': zones,
        'zone_percentages': zone_areas,
        'predominant_damage': max(zone_areas, key=zone_areas.get)
    }
```

---

## 3. LLM Disaster Intelligence Synthesis

### Prompt Templates

#### Flood Situation Assessment

```python
FLOOD_ASSESSMENT_PROMPT = """
You are a disaster management satellite imagery analyst. Based on the following 
satellite-derived flood analysis data for {location}:

FLOOD METRICS:
- Total flooded area: {flooded_area_km2} km²
- Flood coverage: {flood_percentage}% of analyzed area
- Expansion rate: {expansion_rate} km²/day
- Trend: {trend} (expanding/stable/receding)
- NDWI mean value: {ndwi_mean}

INFRASTRUCTURE IMPACT:
- Total structures detected: {total_structures}
- Structures in flood zone: {affected_structures}
- Impact severity: {impact_severity}

TEMPORAL PROGRESSION:
{progression_summary}

PROVIDE:
1. **Situation Assessment** (2-3 sentences summarizing current state)
2. **Escalation Level**: One of [ADVISORY | WARNING | EMERGENCY | CATASTROPHIC]
3. **Evacuation Priority** (which areas/populations should evacuate first)
4. **24-Hour Prediction** (expected changes based on observed trend)
5. **72-Hour Prediction** (projected flood extent if trend continues)
6. **Resource Deployment** (recommended emergency resources and staging areas)

Be specific. Reference the actual numbers from the analysis.
Do not make up data that was not provided.
"""
```

#### Earthquake Damage Report

```python
EARTHQUAKE_ASSESSMENT_PROMPT = """
You are a structural damage assessment specialist using satellite imagery. 
Based on pre/post earthquake comparison for {location}:

DAMAGE METRICS:
- Overall change detected: {change_percentage}%
- Damage severity: {severity}
- Damage zone breakdown:
  - No damage: {no_damage}%
  - Minor: {minor}%
  - Moderate: {moderate}%
  - Severe: {severe}%
  - Destroyed: {destroyed}%

PROVIDE:
1. **Damage Assessment Summary**
2. **Search and Rescue Priorities** (based on damage distribution)
3. **Structural Safety Advisory** (for remaining structures)
4. **Aftershock Risk Advisory** (secondary hazard awareness)
5. **Relief Resource Allocation** (based on damage zones)
"""
```

---

## 4. Output Format

### Flood Analysis Output

```json
{
  "disaster_type": "flood",
  "analysis": {
    "flood_extent": {
      "flooded_area_km2": 42.7,
      "flood_percentage": 18.3,
      "water_mask": "<base64 encoded mask>",
      "detection_method": "NDWI + SAR Fusion"
    },
    "progression": {
      "rate_km2_per_day": 5.2,
      "trend": "expanding",
      "timeline": [
        {"date": "2026-08-22", "area_km2": 12.1},
        {"date": "2026-08-29", "area_km2": 28.4},
        {"date": "2026-09-05", "area_km2": 42.7}
      ]
    },
    "infrastructure_impact": {
      "total_structures": 847,
      "affected_structures": 156,
      "impact_severity": "high"
    },
    "evacuation_zones": {
      "danger_zone_pct": 22.5,
      "warning_zone_pct": 15.3,
      "safe_zone_pct": 62.2,
      "recommended_centers": [...]
    }
  },
  "llm_report": {
    "situation_assessment": "...",
    "escalation_level": "EMERGENCY",
    "evacuation_priority": "...",
    "prediction_24h": "...",
    "prediction_72h": "...",
    "resource_deployment": "..."
  },
  "scene_description": "...",
  "visual_evidence": {
    "rgb_image": "<base64>",
    "flood_overlay": "<base64>",
    "progression_overlay": "<base64>",
    "evacuation_zone_overlay": "<base64>"
  }
}
```
