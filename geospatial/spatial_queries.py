import math
from typing import List, Dict, Any, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SpatialQueryEngine:
    """
    Evaluates spatial queries, spatial radius distance checks, and bounding box intersections for satellite imagery.
    """

    def haversine_distance_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two geographic coordinates in meters.
        """
        R = 6371000.0  # Earth radius in meters
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = math.sin(dlat / 2.0)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return R * c

    def is_within_bounds(self, point_lon: float, point_lat: float, bounds: List[float]) -> bool:
        """Check if geographic point is inside bounding box [min_lon, min_lat, max_lon, max_lat]"""
        min_lon, min_lat, max_lon, max_lat = bounds
        return (min_lon <= point_lon <= max_lon) and (min_lat <= point_lat <= max_lat)

    def evaluate_spatial_filter(
        self,
        target_lat: float,
        target_lon: float,
        query_lat: Optional[float],
        query_lon: Optional[float],
        radius_m: float = 1000.0
    ) -> Dict[str, Any]:
        """
        Evaluate if target feature is within radius_m of query location.
        """
        if query_lat is None or query_lon is None:
            return {'matched': True, 'distance_m': 0.0, 'reason': 'No spatial coordinate filter specified.'}

        dist = self.haversine_distance_m(target_lat, target_lon, query_lat, query_lon)
        matched = dist <= radius_m

        return {
            'matched': matched,
            'distance_m': round(dist, 2),
            'radius_threshold_m': radius_m,
            'reason': f"Feature is {dist:.1f}m from target location (radius limit: {radius_m}m)."
        }
