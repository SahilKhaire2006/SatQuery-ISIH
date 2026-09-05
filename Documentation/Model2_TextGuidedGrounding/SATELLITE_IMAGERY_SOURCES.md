# Real-Time Satellite Imagery Sources — Evaluation & Selection

> **Critical Requirement:** No mock simulation. Every image used for analysis MUST be fetched from a live Earth-observation API.

---

## Source Comparison Matrix

| Source | Data Type | Resolution | Revisit | Cloud Issue? | Cost | Best For |
|---|---|---|---|---|---|---|
| **Sentinel-2 L2A** | Optical RGB + 13 bands | 10m | 5 days | ⚠️ Yes | **Free** | Clear-weather flood mapping, NDWI |
| **Sentinel-1 GRD** | SAR (C-band radar) | 10m | 6 days | ✅ No (sees through clouds) | **Free** | Active flood detection, day/night |
| **Esri World Imagery** | Optical RGB composite | ~1m (varies) | Irregular | ⚠️ Yes | **Free** (no key) | High-res visual, building detection |
| **NASA OPERA DSWx** | Surface water extent | 30m | ~3 days | ✅ Pre-processed | **Free** | Analysis-ready water mask |
| **NASA FIRMS** | Thermal fire hotspots | 375m (VIIRS) | ~3 hours | ✅ Thermal | **Free** | Wildfire detection |
| **Planet (SkySat)** | Optical RGB | 0.5m | Daily | ⚠️ Yes | **Paid** | Ultra-high-res (if budget allows) |

---

## 🥇 Primary: Copernicus Sentinel Hub (via Data Space Ecosystem)

### Why This is the Best Choice

1. **Free & open** — no cost, no rate limits for reasonable usage
2. **Global coverage** — every square meter on Earth
3. **Multi-spectral** — 13 bands including NIR (for NDWI, NDVI computation)
4. **SAR data** — Sentinel-1 penetrates clouds, critical for active flood events
5. **Server-side processing** — evalscripts compute NDWI/NDVI on their servers, minimal local compute
6. **Multi-temporal** — query historical imagery for progression analysis
7. **Well-documented Python library** — `sentinelhub-py`

### Registration & Setup

1. Go to https://dataspace.copernicus.eu/
2. Register a free account
3. Navigate to "Dashboard" → "User Settings" → "OAuth Clients"
4. Create a new OAuth client → get `client_id` and `client_secret`
5. Add to `.env`:
   ```
   COPERNICUS_CLIENT_ID=your_client_id
   COPERNICUS_CLIENT_SECRET=your_client_secret
   ```

### Python Integration

```bash
pip install sentinelhub
```

#### Config Setup
```python
from sentinelhub import SHConfig

config = SHConfig()
config.sh_client_id = os.getenv('COPERNICUS_CLIENT_ID')
config.sh_client_secret = os.getenv('COPERNICUS_CLIENT_SECRET')
config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
```

#### Sentinel-2 True Color Fetch
```python
from sentinelhub import (
    SentinelHubRequest, DataCollection, MimeType, 
    BBox, CRS, bbox_to_dimensions
)

# Evalscript for True Color RGB
evalscript_true_color = """
//VERSION=3
function setup() {
    return { input: ["B04","B03","B02"], output: { bands: 3 } };
}
function evaluatePixel(sample) {
    return [2.5*sample.B04, 2.5*sample.B03, 2.5*sample.B02];
}
"""

bbox = BBox([lon-0.01, lat-0.01, lon+0.01, lat+0.01], crs=CRS.WGS84)
size = bbox_to_dimensions(bbox, resolution=10)

request = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[SentinelHubRequest.input_data(
        data_collection=DataCollection.SENTINEL2_L2A,
        time_interval=('2026-08-01', '2026-09-05'),
        other_args={"dataFilter": {"maxCloudCoverage": 30}}
    )],
    responses=[SentinelHubRequest.output_response('default', MimeType.PNG)],
    bbox=bbox,
    size=size,
    config=config
)

images = request.get_data()  # Returns list of numpy arrays
```

#### NDWI Water Index (Server-Side)
```python
evalscript_ndwi = """
//VERSION=3
function setup() {
    return { input: ["B03","B08"], output: { bands: 1, sampleType: "FLOAT32" } };
}
function evaluatePixel(sample) {
    let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);
    return [ndwi];
}
"""
```

#### Sentinel-1 SAR (Flood Detection Through Clouds)
```python
evalscript_sar = """
//VERSION=3
function setup() {
    return { input: ["VV","VH"], output: { bands: 2, sampleType: "FLOAT32" } };
}
function evaluatePixel(sample) {
    return [sample.VV, sample.VH];
}
"""

request_sar = SentinelHubRequest(
    evalscript=evalscript_sar,
    input_data=[SentinelHubRequest.input_data(
        data_collection=DataCollection.SENTINEL1_IW,
        time_interval=('2026-08-28', '2026-09-05'),
    )],
    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
    bbox=bbox,
    size=size,
    config=config
)
```

---

## 🥈 Secondary/Fallback: Esri World Imagery (Already Integrated)

### Current Implementation

The existing `geospatial/map_fetcher.py` already fetches from Esri:

```python
url = (
    f"https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"
    f"?bbox={bbox_str}&bboxSR=4326&size={w_px},{h_px}&imageSR=4326&format=jpg&f=image"
)
```

### Limitations for Disaster Management

- ❌ No multi-temporal data (only latest composite)
- ❌ No SAR data (cannot see through clouds)
- ❌ No spectral bands (cannot compute NDWI)
- ❌ Unknown acquisition date (composite of multiple dates)
- ✅ Very high resolution in some areas (~1m)
- ✅ No authentication needed
- ✅ Fast response time

### Role in Model 2

- **Fallback** when Sentinel Hub is unavailable
- **Complement** for high-res building/infrastructure detection
- **Visual reference** for disaster reports alongside Sentinel analysis

---

## 🥉 Optional Enhancement: NASA Earthdata

### NASA FIRMS (Fire Information for Resource Management System)

For wildfire disaster scenarios:

```bash
pip install earthaccess
```

```python
import earthaccess

earthaccess.login()  # Uses ~/.netrc or interactive

results = earthaccess.search_data(
    short_name="VNP14IMG",  # VIIRS active fire
    bounding_box=(lon-1, lat-1, lon+1, lat+1),
    temporal=("2026-08-01", "2026-09-05")
)
```

### NASA OPERA DSWx (Dynamic Surface Water Extent)

Pre-computed water masks — no need to threshold NDWI ourselves:

```python
results = earthaccess.search_data(
    short_name="OPERA_L3_DSWX-HLS_V1",
    bounding_box=(lon-0.5, lat-0.5, lon+0.5, lat+0.5),
    temporal=("2026-08-01", "2026-09-05")
)
```

---

## Imagery Source Selection Logic

```python
def select_imagery_source(disaster_type: str, weather: str, temporal_needed: bool) -> str:
    """
    Determine which satellite imagery source to use.
    """
    if disaster_type == 'flood' and weather in ['cloudy', 'rainy', 'storm']:
        return 'sentinel1_sar'     # SAR penetrates clouds
    
    if temporal_needed:
        return 'sentinel2_optical'  # Multi-temporal from Sentinel-2
    
    if disaster_type == 'wildfire':
        return 'nasa_firms'         # Thermal fire detection
    
    # Default to Sentinel-2 optical
    try:
        return 'sentinel2_optical'
    except:
        return 'esri_world_imagery'  # Fallback
```

---

## Rate Limits & Quotas

| Source | Rate Limit | Notes |
|---|---|---|
| Sentinel Hub (free tier) | 5 requests/min, ~5000/month | Sufficient for development & demo |
| Esri World Imagery | No hard limit | Unofficial; be reasonable |
| NASA Earthdata | No hard limit | Requires free account login |

> **Note:** For SIH demonstration purposes, the free Sentinel Hub tier is more than sufficient. Production deployment may need a paid plan for higher throughput.
