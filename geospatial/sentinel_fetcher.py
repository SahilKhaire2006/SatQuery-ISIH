"""
Sentinel Hub / Copernicus Data Space — Real Satellite Imagery Fetcher

Fetches REAL Sentinel-2 (optical RGB, NDWI) and Sentinel-1 (SAR) imagery
from the Copernicus Data Space Ecosystem for disaster management analysis.

Primary source for Model 2 (Text-Guided Grounding for Disaster Management).
Provides multi-temporal imagery for flood progression, earthquake damage, etc.

API Reference: https://documentation.dataspace.copernicus.eu/
Python SDK:    https://sentinelhub-py.readthedocs.io/
"""

import os
import math
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from PIL import Image

from utils.logger import setup_logger

logger = setup_logger(__name__)

# Try importing sentinelhub; gracefully degrade if not installed
try:
    from sentinelhub import (
        SHConfig,
        SentinelHubRequest,
        SentinelHubCatalog,
        DataCollection,
        MimeType,
        BBox,
        CRS,
        bbox_to_dimensions,
        Geometry,
    )
    SENTINELHUB_AVAILABLE = True
except ImportError:
    SENTINELHUB_AVAILABLE = False
    logger.warning(
        "sentinelhub package not installed. "
        "Install with: pip install sentinelhub  "
        "Sentinel Hub features will be unavailable; Esri fallback will be used."
    )

# ─────────────────────────────────────────────────────────────
# Output directory for cached satellite images
# ─────────────────────────────────────────────────────────────
SENTINEL_CACHE_DIR = Path("data/sentinel_cache")
SENTINEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Evalscripts (JavaScript executed server-side on Sentinel Hub)
# ─────────────────────────────────────────────────────────────

EVALSCRIPT_TRUE_COLOR = """
//VERSION=3
function setup() {
    return {
        input: [{bands: ["B04","B03","B02"], units: "DN"}],
        output: {bands: 3, sampleType: "AUTO"}
    };
}
function evaluatePixel(sample) {
    return [2.5 * sample.B04 / 10000,
            2.5 * sample.B03 / 10000,
            2.5 * sample.B02 / 10000];
}
"""

EVALSCRIPT_NDWI = """
//VERSION=3
function setup() {
    return {
        input: [{bands: ["B03","B08"], units: "DN"}],
        output: {bands: 1, sampleType: "FLOAT32"}
    };
}
function evaluatePixel(sample) {
    let green = sample.B03;
    let nir   = sample.B08;
    let ndwi  = (green - nir) / (green + nir + 0.0001);
    return [ndwi];
}
"""

EVALSCRIPT_SAR_VV_VH = """
//VERSION=3
function setup() {
    return {
        input: [{bands: ["VV","VH"], units: "LINEAR_POWER"}],
        output: {bands: 2, sampleType: "FLOAT32"}
    };
}
function evaluatePixel(sample) {
    return [sample.VV, sample.VH];
}
"""

EVALSCRIPT_SAR_RGB_COMPOSITE = """
//VERSION=3
function setup() {
    return {
        input: [{bands: ["VV","VH"], units: "LINEAR_POWER"}],
        output: {bands: 3, sampleType: "AUTO"}
    };
}
function evaluatePixel(sample) {
    // False-color SAR composite: R=VV, G=VH, B=VV/VH
    let vv = Math.sqrt(sample.VV);
    let vh = Math.sqrt(sample.VH);
    let ratio = (sample.VH > 0) ? sample.VV / sample.VH : 0;
    return [3.0 * vv, 8.0 * vh, 2.0 * Math.sqrt(ratio)];
}
"""


# ─────────────────────────────────────────────────────────────
# Sentinel Hub Configuration
# ─────────────────────────────────────────────────────────────

def _get_sentinel_hub_config() -> Optional[Any]:
    """
    Build and return a configured SHConfig object for Copernicus Data Space.
    Returns None if credentials are missing or sentinelhub is not installed.
    """
    if not SENTINELHUB_AVAILABLE:
        return None

    client_id = os.getenv("COPERNICUS_CLIENT_ID", "")
    client_secret = os.getenv("COPERNICUS_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        logger.warning(
            "COPERNICUS_CLIENT_ID / COPERNICUS_CLIENT_SECRET not set in .env. "
            "Sentinel Hub API calls will fail — Esri fallback will be used."
        )
        return None

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_base_url = os.getenv(
        "SENTINEL_HUB_BASE_URL",
        "https://sh.dataspace.copernicus.eu",
    )
    config.sh_token_url = (
        "https://identity.dataspace.copernicus.eu/"
        "auth/realms/CDSE/protocol/openid-connect/token"
    )
    return config


# ─────────────────────────────────────────────────────────────
# Bounding Box Helpers
# ─────────────────────────────────────────────────────────────

def _compute_bbox(
    lat: float, lon: float, area_meters: float = 2000.0
) -> Tuple[List[float], Any]:
    """
    Compute a WGS-84 bounding box centered on (lat, lon) with half-side
    length = area_meters / 2.

    Returns:
        (bbox_coords, BBox_object)
        bbox_coords = [west_lon, south_lat, east_lon, north_lat]
    """
    half_side = area_meters / 2.0
    delta_lat = half_side / 111_320.0
    cos_lat = max(0.1, math.cos(math.radians(lat)))
    delta_lon = delta_lat / cos_lat

    west = lon - delta_lon
    east = lon + delta_lon
    south = lat - delta_lat
    north = lat + delta_lat
    coords = [west, south, east, north]

    if SENTINELHUB_AVAILABLE:
        return coords, BBox(coords, crs=CRS.WGS84)
    return coords, None


def _resolution_to_size(
    bbox_coords: List[float], resolution_m: int = 10
) -> Tuple[int, int]:
    """
    Compute pixel dimensions for the given bbox at the desired resolution.
    """
    if SENTINELHUB_AVAILABLE:
        bbox_obj = BBox(bbox_coords, crs=CRS.WGS84)
        return bbox_to_dimensions(bbox_obj, resolution=resolution_m)
    # Fallback: estimate from bbox extent
    lat_span = bbox_coords[3] - bbox_coords[1]
    lon_span = bbox_coords[2] - bbox_coords[0]
    height = int(lat_span * 111_320 / resolution_m)
    width = int(lon_span * 111_320 * math.cos(math.radians(
        (bbox_coords[1] + bbox_coords[3]) / 2)) / resolution_m)
    return max(width, 64), max(height, 64)


# ─────────────────────────────────────────────────────────────
# Core Fetcher Functions
# ─────────────────────────────────────────────────────────────

def fetch_sentinel2_rgb(
    lat: float,
    lon: float,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    area_meters: float = 2000.0,
    max_cloud_pct: int = 40,
    resolution_m: int = 10,
) -> Dict[str, Any]:
    """
    Fetch a Sentinel-2 L2A True Color RGB image for the region.

    Args:
        lat, lon: Center coordinates (WGS-84).
        date_from, date_to: ISO date strings (YYYY-MM-DD). Defaults to last 30 days.
        area_meters: Side length of the square bounding box in meters.
        max_cloud_pct: Maximum cloud-cover percentage filter.
        resolution_m: Pixel resolution in meters (default 10m).

    Returns:
        Dict with keys: image (np.ndarray RGB), bbox, timestamp, source, metadata.
    """
    config = _get_sentinel_hub_config()
    if config is None:
        raise RuntimeError(
            "Sentinel Hub not available (missing credentials or package). "
            "Use imagery_router to fall back to Esri."
        )

    now = datetime.utcnow()
    if date_to is None:
        date_to = now.strftime("%Y-%m-%d")
    if date_from is None:
        date_from = (now - timedelta(days=30)).strftime("%Y-%m-%d")

    bbox_coords, bbox_obj = _compute_bbox(lat, lon, area_meters)
    size = _resolution_to_size(bbox_coords, resolution_m)

    logger.info(
        f"Sentinel-2 RGB request: ({lat:.5f}, {lon:.5f}), "
        f"area={area_meters}m, dates={date_from}→{date_to}, "
        f"cloud<={max_cloud_pct}%, size={size}"
    )

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_TRUE_COLOR,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(date_from, date_to),
                other_args={"dataFilter": {"maxCloudCoverage": max_cloud_pct}},
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", MimeType.PNG)
        ],
        bbox=bbox_obj,
        size=size,
        config=config,
    )

    images = request.get_data()

    if not images or images[0] is None:
        raise RuntimeError(
            f"No Sentinel-2 imagery found for ({lat}, {lon}) "
            f"in date range {date_from} to {date_to} with cloud<={max_cloud_pct}%."
        )

    img_arr = np.array(images[0])
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)

    # Ensure uint8
    if img_arr.dtype == np.float64 or img_arr.dtype == np.float32:
        img_arr = np.clip(img_arr * 255, 0, 255).astype(np.uint8)

    # Cache to disk
    cache_name = (
        f"s2_rgb_{lat:.4f}_{lon:.4f}_{date_to}.png"
    )
    cache_path = SENTINEL_CACHE_DIR / cache_name
    Image.fromarray(img_arr).save(cache_path)
    logger.info(f"Sentinel-2 RGB cached → {cache_path}")

    return {
        "image": img_arr,
        "image_path": str(cache_path),
        "bbox": bbox_coords,
        "center_coords": [lat, lon],
        "area_meters": area_meters,
        "date_range": [date_from, date_to],
        "resolution_m": resolution_m,
        "source": "Copernicus Sentinel-2 L2A",
        "type": "optical_rgb",
    }


def fetch_sentinel1_sar(
    lat: float,
    lon: float,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    area_meters: float = 2000.0,
    resolution_m: int = 10,
    return_rgb_composite: bool = True,
) -> Dict[str, Any]:
    """
    Fetch Sentinel-1 IW GRD SAR imagery.

    SAR is critical for flood detection because radar penetrates clouds and
    works day/night — essential during active storm/flood events.

    Args:
        lat, lon: Center coordinates (WGS-84).
        date_from, date_to: ISO date strings.
        area_meters: Side length of the bounding box in meters.
        resolution_m: Pixel resolution in meters.
        return_rgb_composite: If True, returns a false-color RGB SAR composite.
                              If False, returns raw VV+VH bands as float32.

    Returns:
        Dict with keys: image, bbox, timestamp, source, metadata.
    """
    config = _get_sentinel_hub_config()
    if config is None:
        raise RuntimeError(
            "Sentinel Hub not available (missing credentials or package)."
        )

    now = datetime.utcnow()
    if date_to is None:
        date_to = now.strftime("%Y-%m-%d")
    if date_from is None:
        date_from = (now - timedelta(days=15)).strftime("%Y-%m-%d")

    bbox_coords, bbox_obj = _compute_bbox(lat, lon, area_meters)
    size = _resolution_to_size(bbox_coords, resolution_m)

    evalscript = (
        EVALSCRIPT_SAR_RGB_COMPOSITE if return_rgb_composite
        else EVALSCRIPT_SAR_VV_VH
    )
    mime = MimeType.PNG if return_rgb_composite else MimeType.TIFF

    logger.info(
        f"Sentinel-1 SAR request: ({lat:.5f}, {lon:.5f}), "
        f"area={area_meters}m, dates={date_from}→{date_to}, size={size}"
    )

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL1_IW,
                time_interval=(date_from, date_to),
                other_args={
                    "processing": {
                        "backCoeff": "SIGMA0_ELLIPSOID",
                        "orthorectify": True,
                    }
                },
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", mime)
        ],
        bbox=bbox_obj,
        size=size,
        config=config,
    )

    images = request.get_data()

    if not images or images[0] is None:
        raise RuntimeError(
            f"No Sentinel-1 SAR imagery found for ({lat}, {lon}) "
            f"in date range {date_from} to {date_to}."
        )

    img_arr = np.array(images[0])

    # For RGB composite: ensure uint8 for display
    if return_rgb_composite and img_arr.dtype in (np.float32, np.float64):
        img_arr = np.clip(img_arr * 255, 0, 255).astype(np.uint8)

    cache_name = f"s1_sar_{lat:.4f}_{lon:.4f}_{date_to}.png"
    cache_path = SENTINEL_CACHE_DIR / cache_name
    if img_arr.ndim == 3 and img_arr.shape[2] in (3, 4):
        Image.fromarray(img_arr[:, :, :3]).save(cache_path)
    else:
        np.save(cache_path.with_suffix(".npy"), img_arr)
    logger.info(f"Sentinel-1 SAR cached → {cache_path}")

    return {
        "image": img_arr,
        "image_path": str(cache_path),
        "bbox": bbox_coords,
        "center_coords": [lat, lon],
        "area_meters": area_meters,
        "date_range": [date_from, date_to],
        "resolution_m": resolution_m,
        "source": "Copernicus Sentinel-1 IW GRD",
        "type": "sar",
        "polarization": "VV+VH",
    }


def fetch_ndwi_layer(
    lat: float,
    lon: float,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    area_meters: float = 2000.0,
    resolution_m: int = 10,
) -> Dict[str, Any]:
    """
    Fetch a server-side computed NDWI (Normalized Difference Water Index) map.

    NDWI = (Green − NIR) / (Green + NIR)
    Values > ~0.3 indicate water surfaces.

    This is computed entirely on Sentinel Hub servers — no local band math needed.

    Returns:
        Dict with 'ndwi_map' (float32 ndarray, values in [-1, 1]), plus metadata.
    """
    config = _get_sentinel_hub_config()
    if config is None:
        raise RuntimeError(
            "Sentinel Hub not available (missing credentials or package)."
        )

    now = datetime.utcnow()
    if date_to is None:
        date_to = now.strftime("%Y-%m-%d")
    if date_from is None:
        date_from = (now - timedelta(days=30)).strftime("%Y-%m-%d")

    bbox_coords, bbox_obj = _compute_bbox(lat, lon, area_meters)
    size = _resolution_to_size(bbox_coords, resolution_m)

    logger.info(
        f"NDWI request: ({lat:.5f}, {lon:.5f}), "
        f"area={area_meters}m, dates={date_from}→{date_to}"
    )

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_NDWI,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(date_from, date_to),
                other_args={"dataFilter": {"maxCloudCoverage": 40}},
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", MimeType.TIFF)
        ],
        bbox=bbox_obj,
        size=size,
        config=config,
    )

    images = request.get_data()

    if not images or images[0] is None:
        raise RuntimeError(
            f"No NDWI data available for ({lat}, {lon}) "
            f"in date range {date_from} to {date_to}."
        )

    ndwi_map = np.array(images[0], dtype=np.float32)
    if ndwi_map.ndim == 3:
        ndwi_map = ndwi_map[:, :, 0]  # single-band

    # Cache
    cache_name = f"ndwi_{lat:.4f}_{lon:.4f}_{date_to}.npy"
    cache_path = SENTINEL_CACHE_DIR / cache_name
    np.save(cache_path, ndwi_map)
    logger.info(f"NDWI map cached → {cache_path}  (shape={ndwi_map.shape})")

    return {
        "ndwi_map": ndwi_map,
        "image_path": str(cache_path),
        "bbox": bbox_coords,
        "center_coords": [lat, lon],
        "area_meters": area_meters,
        "date_range": [date_from, date_to],
        "resolution_m": resolution_m,
        "source": "Copernicus Sentinel-2 L2A (NDWI)",
        "type": "ndwi",
    }


def fetch_multi_temporal(
    lat: float,
    lon: float,
    lookback_days: int = 14,
    num_snapshots: int = 3,
    area_meters: float = 2000.0,
    resolution_m: int = 10,
    max_cloud_pct: int = 50,
    include_ndwi: bool = True,
) -> Dict[str, Any]:
    """
    Fetch multiple temporally-spaced Sentinel-2 images for the same region.
    Used for flood progression analysis and change detection.

    Strategy: Divide the lookback window into `num_snapshots` equal intervals,
    fetch the best-available image for each interval.

    Args:
        lat, lon: Center coordinates.
        lookback_days: How many days back to look.
        num_snapshots: Number of temporal snapshots to retrieve.
        area_meters: Side length of bounding box.
        resolution_m: Pixel resolution.
        max_cloud_pct: Maximum cloud cover filter.
        include_ndwi: Also fetch NDWI map for each snapshot.

    Returns:
        Dict with 'snapshots' list, each containing image + ndwi_map + dates.
    """
    config = _get_sentinel_hub_config()
    if config is None:
        raise RuntimeError(
            "Sentinel Hub not available (missing credentials or package)."
        )

    now = datetime.utcnow()
    interval_days = max(1, lookback_days // num_snapshots)

    snapshots = []

    for i in range(num_snapshots):
        # Work backwards from today
        end_date = now - timedelta(days=i * interval_days)
        start_date = end_date - timedelta(days=interval_days)

        date_from = start_date.strftime("%Y-%m-%d")
        date_to = end_date.strftime("%Y-%m-%d")

        logger.info(
            f"Multi-temporal snapshot {i+1}/{num_snapshots}: "
            f"{date_from} → {date_to}"
        )

        snapshot = {"date_from": date_from, "date_to": date_to, "index": i}

        # Fetch RGB
        try:
            rgb_result = fetch_sentinel2_rgb(
                lat, lon,
                date_from=date_from,
                date_to=date_to,
                area_meters=area_meters,
                max_cloud_pct=max_cloud_pct,
                resolution_m=resolution_m,
            )
            snapshot["rgb"] = rgb_result
        except Exception as e:
            logger.warning(f"RGB snapshot {i+1} failed: {e}")
            snapshot["rgb"] = None

        # Fetch NDWI if requested
        if include_ndwi:
            try:
                ndwi_result = fetch_ndwi_layer(
                    lat, lon,
                    date_from=date_from,
                    date_to=date_to,
                    area_meters=area_meters,
                    resolution_m=resolution_m,
                )
                snapshot["ndwi"] = ndwi_result
            except Exception as e:
                logger.warning(f"NDWI snapshot {i+1} failed: {e}")
                snapshot["ndwi"] = None

        snapshots.append(snapshot)

    # Reverse so oldest comes first (chronological order)
    snapshots.reverse()

    successful = sum(1 for s in snapshots if s.get("rgb") is not None)
    logger.info(
        f"Multi-temporal fetch complete: {successful}/{num_snapshots} "
        f"successful snapshots for ({lat:.4f}, {lon:.4f})"
    )

    return {
        "snapshots": snapshots,
        "center_coords": [lat, lon],
        "area_meters": area_meters,
        "lookback_days": lookback_days,
        "successful_count": successful,
        "source": "Copernicus Sentinel-2 L2A (multi-temporal)",
    }


# ─────────────────────────────────────────────────────────────
# Catalog Search (optional — find exact acquisition dates)
# ─────────────────────────────────────────────────────────────

def search_sentinel_catalog(
    lat: float,
    lon: float,
    date_from: str,
    date_to: str,
    collection: str = "sentinel-2-l2a",
    max_cloud_pct: int = 30,
) -> List[Dict]:
    """
    Search the Sentinel Hub catalog for available acquisitions in a region.
    Useful for finding exact dates before fetching imagery.
    """
    config = _get_sentinel_hub_config()
    if config is None:
        return []

    bbox_coords, bbox_obj = _compute_bbox(lat, lon, area_meters=5000)

    data_collection = (
        DataCollection.SENTINEL2_L2A
        if "sentinel-2" in collection.lower()
        else DataCollection.SENTINEL1_IW
    )

    try:
        catalog = SentinelHubCatalog(config=config)
        search_results = catalog.search(
            data_collection,
            bbox=bbox_obj,
            time=(date_from, date_to),
            fields={
                "include": [
                    "id",
                    "properties.datetime",
                    "properties.eo:cloud_cover",
                ],
                "exclude": [],
            },
        )

        results = []
        for item in search_results:
            props = item.get("properties", {})
            cloud = props.get("eo:cloud_cover", 100)
            if cloud <= max_cloud_pct:
                results.append({
                    "id": item.get("id"),
                    "datetime": props.get("datetime"),
                    "cloud_cover": cloud,
                })

        results.sort(key=lambda x: x.get("cloud_cover", 100))
        logger.info(
            f"Catalog search: {len(results)} acquisitions found "
            f"for ({lat}, {lon}) between {date_from} and {date_to}"
        )
        return results

    except Exception as e:
        logger.error(f"Sentinel catalog search failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# Utility: Check Sentinel Hub availability
# ─────────────────────────────────────────────────────────────

def is_sentinel_hub_available() -> bool:
    """Quick check if Sentinel Hub is configured and reachable."""
    if not SENTINELHUB_AVAILABLE:
        return False
    config = _get_sentinel_hub_config()
    return config is not None
