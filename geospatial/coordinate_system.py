import math
from typing import Tuple, Dict, Any, List
from utils.logger import setup_logger

logger = setup_logger(__name__)


class CoordinateTransformer:
    """
    Handles coordinate reference system (CRS) transformations and pixel-to-lat/lon mappings.
    """

    def pixel_to_latlon(
        self,
        pixel_x: int,
        pixel_y: int,
        img_width: int,
        img_height: int,
        bounds: List[float]
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinate (x, y) to geographic (longitude, latitude) within bounds [min_lon, min_lat, max_lon, max_lat].
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        lon_ratio = pixel_x / float(max(1, img_width))
        lat_ratio = pixel_y / float(max(1, img_height))

        lon = min_lon + lon_ratio * (max_lon - min_lon)
        lat = max_lat - lat_ratio * (max_lat - min_lat)  # Inverse Y axis for raster

        return round(lon, 6), round(lat, 6)

    def latlon_to_pixel(
        self,
        lon: float,
        lat: float,
        img_width: int,
        img_height: int,
        bounds: List[float]
    ) -> Tuple[int, int]:
        """
        Convert geographic coordinate (longitude, latitude) to image pixel coordinate (x, y).
        """
        min_lon, min_lat, max_lon, max_lat = bounds

        x = int(((lon - min_lon) / (max_lon - min_lon + 1e-7)) * img_width)
        y = int(((max_lat - lat) / (max_lat - min_lat + 1e-7)) * img_height)

        x = max(0, min(img_width - 1, x))
        y = max(0, min(img_height - 1, y))

        return x, y

    def epsg4326_to_epsg3857(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        Convert WGS84 (EPSG:4326) Lat/Lon to Web Mercator (EPSG:3857) meters.
        """
        x = lon * 20037508.34 / 180.0
        y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
        y = y * 20037508.34 / 180.0
        return round(x, 2), round(y, 2)
