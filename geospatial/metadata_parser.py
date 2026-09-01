import json
from typing import Dict, Any, Optional, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)


class GeoMetadataParser:
    """
    Parses spatial metadata from GeoTIFF headers, JSON metadata strings, and geographic tags.
    """

    def parse_metadata(self, geo_metadata: Optional[str]) -> Dict[str, Any]:
        """
        Parse raw geospatial metadata input string into structured spatial dictionary.
        """
        if not geo_metadata:
            # Default fallback spatial metadata (WGS84)
            return {
                'has_geospatial': False,
                'crs': 'EPSG:4326',
                'bounds': [77.1000, 28.5000, 77.3000, 28.7000],  # [min_lon, min_lat, max_lon, max_lat]
                'resolution_m': 10.0,
                'pixel_scale': [0.0001, -0.0001],
                'origin': [77.1000, 28.7000]
            }

        try:
            if isinstance(geo_metadata, str):
                parsed = json.loads(geo_metadata)
            else:
                parsed = dict(geo_metadata)

            crs = parsed.get('crs', 'EPSG:4326')
            bounds = parsed.get('bounds', [77.1000, 28.5000, 77.3000, 28.7000])
            resolution = float(parsed.get('resolution_m', 10.0))

            return {
                'has_geospatial': True,
                'crs': crs,
                'bounds': bounds,
                'resolution_m': resolution,
                'center_lat': round((bounds[1] + bounds[3]) / 2.0, 4),
                'center_lon': round((bounds[0] + bounds[2]) / 2.0, 4),
                'raw_metadata': parsed
            }
        except Exception as e:
            logger.warning(f"Error parsing geospatial metadata string: {e}. Using standard CRS bounds.")
            return {
                'has_geospatial': False,
                'crs': 'EPSG:4326',
                'bounds': [77.1000, 28.5000, 77.3000, 28.7000],
                'resolution_m': 10.0,
                'error': str(e)
            }
