"""
Automated Geospatial Map Fetcher & Geocoder

Fetches 500m x 500m (or custom size) high-resolution satellite imagery for any given coordinates,
address, or place name using OpenStreetMap Nominatim geocoding and Esri World Imagery REST API.
"""

import math
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

from utils.logger import setup_logger

logger = setup_logger(__name__)

FETCHED_MAPS_DIR = Path("text_guided_grounding/data/fetched_maps")


def geocode_location(location_query: str) -> Tuple[float, float, str]:
    """
    Geocode an address, place name, or lat/lon coordinate string from query text.

    Args:
        location_query: Address, landmark name, "lat, lon" string, or natural language query

    Returns:
        Tuple of (latitude, longitude, display_address_name)
    """
    # Clean up input string
    cleaned_query = location_query.strip().rstrip(".").strip()

    # Strip noise prefix words like "TARGET:", "query:", "find", "locate"
    for noise_prefix in ["target:", "query:", "locate", "find", "search", "detect"]:
        if cleaned_query.lower().startswith(noise_prefix):
            cleaned_query = cleaned_query[len(noise_prefix):].strip()

    # 1. Check if query contains raw numerical coordinates "lat, lon"
    coords = cleaned_query.replace(";", ",").split(",")
    if len(coords) == 2:
        try:
            lat = float(coords[0].strip())
            lon = float(coords[1].strip())
            return lat, lon, f"Coordinates ({lat:.4f}, {lon:.4f})"
        except ValueError:
            pass

    # Build list of query candidates (extracted place candidate first, then full cleaned query)
    candidates = []

    # Extract location after location prepositions: " near ", " in ", " at ", " around "
    lower_query = cleaned_query.lower()
    for prep in [" near ", " in ", " at ", " around ", " located in ", " located at ", " located near "]:
        if prep in lower_query:
            idx = lower_query.find(prep)
            extracted = cleaned_query[idx + len(prep):].strip().rstrip(".").strip()
            if extracted and len(extracted) >= 3:
                candidates.append(extracted)
                break

    candidates.append(cleaned_query)

    # Generate fallback location variations (e.g. Katraj, Pune or Sukhsagar Nagar, Pune)
    extended_candidates = list(candidates)
    for cand in candidates:
        parts = [p.strip() for p in cand.replace(",", " ").split() if p.strip()]
        if len(parts) >= 2:
            extended_candidates.append(", ".join(parts))
            extended_candidates.append(f"{parts[-2]}, {parts[-1]}")
            extended_candidates.append(f"{parts[0]}, {parts[-1]}")
            extended_candidates.append(parts[-1])

    # Deduplicate candidates while preserving order
    seen = set()
    final_candidates = []
    for c in extended_candidates:
        if c.lower() not in seen and len(c) >= 3:
            seen.add(c.lower())
            final_candidates.append(c)

    # 2. Try Nominatim Geocoding on candidates
    for candidate in final_candidates:
        encoded_q = urllib.parse.quote(candidate)
        url = f"https://nominatim.openstreetmap.org/search?q={encoded_q}&format=json&limit=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'SatQuery-ISIH/1.0'})

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if data:
                    lat = float(data[0]['lat'])
                    lon = float(data[0]['lon'])
                    display_name = data[0].get('display_name', candidate)
                    logger.info(f"Geocoded '{candidate}' -> ({lat}, {lon}): {display_name}")
                    return lat, lon, display_name
        except Exception as e:
            logger.error(f"Geocoding request failed for '{candidate}': {e}")

    raise ValueError(f"Could not resolve location for query '{location_query}'. Please provide valid coordinates or location name.")


def fetch_satellite_image_tile(
    lat: float,
    lon: float,
    area_meters: float = 500.0,
    output_dir: Path = FETCHED_MAPS_DIR,
    image_size: Tuple[int, int] = (1024, 1024)
) -> Dict[str, Any]:
    """
    Fetch a satellite image tile for an area_meters x area_meters square (default 500m x 500m)
    centered at (lat, lon) using Esri World Imagery REST API.

    Args:
        lat: Latitude center float
        lon: Longitude center float
        area_meters: Width & height of the bounding square in meters (default 500.0)
        output_dir: Directory to store fetched map tile
        image_size: Output image width and height tuple (default 1024x1024)

    Returns:
        Dict containing:
            - 'image_path': Path to saved satellite image
            - 'image': Loaded RGB numpy array
            - 'bbox_geo': (xmin_lon, ymin_lat, xmax_lon, ymax_lat)
            - 'center_coords': (lat, lon)
            - 'area_meters': float
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"sat_{int(area_meters)}m_{lat:.5f}_{lon:.5f}.jpg"
    out_path = output_dir / filename

    # Calculate bounding box offset in degrees for area_meters x area_meters
    # Half side length in meters
    half_side = area_meters / 2.0
    # 1 deg latitude ≈ 111,320 meters
    delta_lat = half_side / 111320.0
    cos_lat = max(0.1, math.cos(math.radians(lat)))
    delta_lon = delta_lat / cos_lat

    ymin_lat = lat - delta_lat
    ymax_lat = lat + delta_lat
    xmin_lon = lon - delta_lon
    xmax_lon = lon + delta_lon

    bbox_str = f"{xmin_lon:.6f},{ymin_lat:.6f},{xmax_lon:.6f},{ymax_lat:.6f}"
    w_px, h_px = image_size

    url = (
        f"https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"
        f"?bbox={bbox_str}&bboxSR=4326&size={w_px},{h_px}&imageSR=4326&format=jpg&f=image"
    )

    req = urllib.request.Request(url, headers={'User-Agent': 'SatQuery-ISIH/1.0'})

    try:
        with urllib.request.urlopen(req, timeout=15) as resp, open(out_path, "wb") as f:
            f.write(resp.read())

        pil_img = Image.open(out_path).convert("RGB")
        img_arr = np.array(pil_img)
        logger.info(f"Fetched {area_meters:.0f}m x {area_meters:.0f}m satellite image for ({lat:.5f}, {lon:.5f}) -> {out_path}")

        return {
            "image_path": str(out_path),
            "image": img_arr,
            "bbox_geo": [xmin_lon, ymin_lat, xmax_lon, ymax_lat],
            "center_coords": [lat, lon],
            "area_meters": area_meters,
            "image_size": [w_px, h_px]
        }

    except Exception as e:
        logger.error(f"Failed to fetch satellite imagery tile for ({lat}, {lon}): {e}")
        raise RuntimeError(f"Satellite map tile fetch error: {e}")


def fetch_1km_satellite_image(lat: float, lon: float, **kwargs) -> Dict[str, Any]:
    """Alias for backwards compatibility fetching 1000m x 1000m tile."""
    return fetch_satellite_image_tile(lat=lat, lon=lon, area_meters=1000.0, **kwargs)
