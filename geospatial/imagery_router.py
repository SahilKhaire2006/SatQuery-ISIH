"""
Imagery Router — Smart Satellite Source Selection

Decides which satellite imagery source to use based on:
- Disaster type (flood vs earthquake vs general)
- Weather conditions (SAR for clouds/rain, optical for clear weather)
- Temporal needs (multi-temporal for progression analysis)
- Availability (fallback from Sentinel Hub → Esri World Imagery)

This module ensures Model 2 always gets REAL satellite imagery,
never a mock or placeholder.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageryRouter:
    """
    Routes satellite imagery requests to the optimal source.
    Handles fallback logic when primary sources are unavailable.
    """

    def __init__(self):
        # Check which providers are available
        self._sentinel_available = self._check_sentinel_hub()
        self._nasa_available = True  # NASA FIRMS open endpoint is available
        self._esri_available = True  # Always available (no auth needed)

        providers = []
        if self._sentinel_available:
            providers.append("Sentinel Hub")
        if self._nasa_available:
            providers.append("NASA Earthdata & FIRMS")
        if self._esri_available:
            providers.append("Esri World Imagery")

        logger.info(
            f"ImageryRouter initialized. Available providers: {providers}"
        )

    @staticmethod
    def _check_sentinel_hub() -> bool:
        """Check if Sentinel Hub is configured."""
        try:
            from geospatial.sentinel_fetcher import is_sentinel_hub_available
            return is_sentinel_hub_available()
        except Exception:
            return False

    def fetch_disaster_imagery(
        self,
        lat: float,
        lon: float,
        disaster_type: str = "flood",
        area_meters: float = 2000.0,
        lookback_days: int = 14,
        include_sar: bool = True,
        include_multi_temporal: bool = True,
    ) -> Dict[str, Any]:
        """
        Master entry point: fetch all relevant imagery for a disaster analysis.

        This function orchestrates multiple fetch calls based on the disaster
        type and returns a comprehensive imagery bundle.

        Args:
            lat, lon: Center coordinates.
            disaster_type: 'flood', 'earthquake', 'wildfire', or 'general'.
            area_meters: Side length of bounding box (meters). Wider for disasters.
            lookback_days: How many days back for temporal analysis.
            include_sar: Whether to also fetch SAR data (recommended for floods).
            include_multi_temporal: Whether to fetch temporal snapshots.

        Returns:
            Dict with imagery bundle:
                primary_image: np.ndarray (RGB for display/analysis)
                sar_image: np.ndarray or None
                ndwi_map: np.ndarray or None
                temporal_snapshots: list or None
                metadata: source info, dates, bbox, etc.
        """
        logger.info(
            f"Imagery request: disaster={disaster_type}, "
            f"location=({lat:.4f}, {lon:.4f}), area={area_meters}m"
        )

        result = {
            "primary_image": None,
            "sar_image": None,
            "ndwi_map": None,
            "temporal_snapshots": None,
            "metadata": {
                "center_coords": [lat, lon],
                "area_meters": area_meters,
                "disaster_type": disaster_type,
                "sources_used": [],
                "fetch_timestamp": datetime.utcnow().isoformat(),
            },
        }

        # --- Strategy selection based on disaster type ---

        if disaster_type == "flood":
            result = self._fetch_flood_imagery(
                lat, lon, area_meters, lookback_days,
                include_sar, include_multi_temporal, result,
            )

        elif disaster_type == "earthquake":
            result = self._fetch_earthquake_imagery(
                lat, lon, area_meters, lookback_days, result,
            )

        elif disaster_type == "wildfire":
            result = self._fetch_wildfire_imagery(
                lat, lon, area_meters, lookback_days, result,
            )

        else:
            # General disaster or unknown type — fetch optical + NDWI
            result = self._fetch_general_imagery(
                lat, lon, area_meters, result,
            )

        # Fallback: if primary image is still None, use Esri
        if result["primary_image"] is None:
            logger.warning(
                "All Sentinel fetches failed. Falling back to Esri World Imagery."
            )
            result = self._fetch_esri_fallback(lat, lon, area_meters, result)

        if result["primary_image"] is None:
            logger.error("All imagery sources failed. No image available.")

        return result

    # ─────────────────────────────────────────────
    # Disaster-specific fetch strategies
    # ─────────────────────────────────────────────

    def _fetch_flood_imagery(
        self, lat, lon, area_meters, lookback_days,
        include_sar, include_multi_temporal, result,
    ) -> Dict:
        """
        Flood-optimized fetch strategy:
        1. Sentinel-2 RGB (current view)
        2. NDWI water index (water extent)
        3. Sentinel-1 SAR (cloud-penetrating flood map)
        4. Multi-temporal series (flood progression)
        """
        # 1. Current optical RGB
        if self._sentinel_available:
            try:
                from geospatial.sentinel_fetcher import fetch_sentinel2_rgb
                rgb_result = fetch_sentinel2_rgb(
                    lat, lon, area_meters=area_meters, max_cloud_pct=50,
                )
                result["primary_image"] = rgb_result["image"]
                result["metadata"]["sources_used"].append(rgb_result["source"])
                result["metadata"]["primary_dates"] = rgb_result["date_range"]
                result["metadata"]["primary_path"] = rgb_result["image_path"]
                logger.info("[OK] Sentinel-2 RGB fetched successfully")
            except Exception as e:
                logger.warning(f"Sentinel-2 RGB fetch failed: {e}")

        # 2. NDWI water index
        if self._sentinel_available:
            try:
                from geospatial.sentinel_fetcher import fetch_ndwi_layer
                ndwi_result = fetch_ndwi_layer(
                    lat, lon, area_meters=area_meters,
                )
                result["ndwi_map"] = ndwi_result["ndwi_map"]
                result["metadata"]["sources_used"].append("Sentinel-2 NDWI")
                logger.info("[OK] NDWI water index fetched successfully")
            except Exception as e:
                logger.warning(f"NDWI fetch failed: {e}")

        # 3. SAR (critical for active floods -- works through clouds/rain)
        if include_sar and self._sentinel_available:
            try:
                from geospatial.sentinel_fetcher import fetch_sentinel1_sar
                sar_result = fetch_sentinel1_sar(
                    lat, lon, area_meters=area_meters,
                    return_rgb_composite=True,
                )
                result["sar_image"] = sar_result["image"]
                result["metadata"]["sources_used"].append(sar_result["source"])
                result["metadata"]["sar_path"] = sar_result["image_path"]
                logger.info("[OK] Sentinel-1 SAR fetched successfully")
            except Exception as e:
                logger.warning(f"Sentinel-1 SAR fetch failed: {e}")

        # 4. Multi-temporal snapshots (flood progression)
        if include_multi_temporal and self._sentinel_available:
            try:
                from geospatial.sentinel_fetcher import fetch_multi_temporal
                temporal_result = fetch_multi_temporal(
                    lat, lon,
                    lookback_days=lookback_days,
                    num_snapshots=3,
                    area_meters=area_meters,
                    include_ndwi=True,
                )
                result["temporal_snapshots"] = temporal_result["snapshots"]
                result["metadata"]["temporal_count"] = temporal_result[
                    "successful_count"
                ]
                logger.info(
                    f"[OK] Multi-temporal: {temporal_result['successful_count']} "
                    f"snapshots fetched"
                )
            except Exception as e:
                logger.warning(f"Multi-temporal fetch failed: {e}")

        return result

    def _fetch_earthquake_imagery(
        self, lat, lon, area_meters, lookback_days, result,
    ) -> Dict:
        """
        Earthquake-optimized fetch:
        1. Post-event optical (current)
        2. Pre-event optical (before earthquake)
        Used for change detection / damage assessment.
        """
        now = datetime.utcnow()

        if self._sentinel_available:
            # Post-event (recent)
            try:
                from geospatial.sentinel_fetcher import fetch_sentinel2_rgb
                post_result = fetch_sentinel2_rgb(
                    lat, lon, area_meters=area_meters, max_cloud_pct=50,
                )
                result["primary_image"] = post_result["image"]
                result["metadata"]["sources_used"].append(
                    post_result["source"] + " (post-event)"
                )
                result["metadata"]["post_event_dates"] = post_result["date_range"]
                logger.info("[OK] Post-event optical fetched")
            except Exception as e:
                logger.warning(f"Post-event fetch failed: {e}")

            # Pre-event (before the lookback window)
            try:
                pre_end = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
                pre_start = (now - timedelta(days=lookback_days + 30)).strftime(
                    "%Y-%m-%d"
                )
                pre_result = fetch_sentinel2_rgb(
                    lat, lon,
                    date_from=pre_start,
                    date_to=pre_end,
                    area_meters=area_meters,
                    max_cloud_pct=40,
                )
                result["pre_event_image"] = pre_result["image"]
                result["metadata"]["sources_used"].append(
                    pre_result["source"] + " (pre-event)"
                )
                result["metadata"]["pre_event_dates"] = pre_result["date_range"]
                logger.info("[OK] Pre-event optical fetched")
            except Exception as e:
                logger.warning(f"Pre-event fetch failed: {e}")

        return result

    def _fetch_wildfire_imagery(
        self, lat, lon, area_meters, lookback_days, result,
    ) -> Dict:
        """
        Wildfire-optimized fetch:
        1. NASA FIRMS Active Fire / Thermal Hotspot Detection (VIIRS/MODIS)
        2. Latest optical RGB (burn scar mapping)
        """
        # 1. Query NASA FIRMS Active Fire Hotspots
        try:
            from geospatial.nasa_earthdata_fetcher import fetch_nasa_firms_hotspots
            firms_data = fetch_nasa_firms_hotspots(lat, lon, day_range=min(lookback_days, 10))
            result["firms_hotspots"] = firms_data
            result["metadata"]["sources_used"].append(firms_data["source"])
            logger.info(f"[OK] NASA FIRMS active fire query returned {firms_data['hotspot_count']} hotspots")
        except Exception as e:
            logger.warning(f"NASA FIRMS query notice: {e}")

        # 2. Sentinel-2 Optical RGB for burn scar mapping
        if self._sentinel_available:
            try:
                from geospatial.sentinel_fetcher import fetch_sentinel2_rgb
                rgb_result = fetch_sentinel2_rgb(
                    lat, lon, area_meters=area_meters, max_cloud_pct=60,
                )
                result["primary_image"] = rgb_result["image"]
                result["metadata"]["sources_used"].append(rgb_result["source"])
                logger.info("[OK] Wildfire optical fetched")
            except Exception as e:
                logger.warning(f"Wildfire optical fetch failed: {e}")

        return result

    def _fetch_general_imagery(
        self, lat, lon, area_meters, result,
    ) -> Dict:
        """
        General disaster: fetch the best available optical + NDWI.
        """
        if self._sentinel_available:
            try:
                from geospatial.sentinel_fetcher import fetch_sentinel2_rgb
                rgb_result = fetch_sentinel2_rgb(
                    lat, lon, area_meters=area_meters, max_cloud_pct=50,
                )
                result["primary_image"] = rgb_result["image"]
                result["metadata"]["sources_used"].append(rgb_result["source"])
                logger.info("[OK] General optical fetched")
            except Exception as e:
                logger.warning(f"General optical fetch failed: {e}")

            try:
                from geospatial.sentinel_fetcher import fetch_ndwi_layer
                ndwi_result = fetch_ndwi_layer(
                    lat, lon, area_meters=area_meters,
                )
                result["ndwi_map"] = ndwi_result["ndwi_map"]
                result["metadata"]["sources_used"].append("Sentinel-2 NDWI")
            except Exception as e:
                logger.warning(f"NDWI fetch failed: {e}")

        return result

    # ─────────────────────────────────────────────
    # Esri World Imagery Fallback
    # ─────────────────────────────────────────────

    def _fetch_esri_fallback(
        self, lat: float, lon: float, area_meters: float, result: Dict,
    ) -> Dict:
        """
        Fallback to the existing Esri World Imagery fetcher.
        This always works (no authentication needed) but provides only
        a single optical composite -- no SAR, no multi-temporal, no NDWI.
        """
        try:
            from geospatial.map_fetcher import fetch_satellite_image_tile
            tile_result = fetch_satellite_image_tile(
                lat=lat, lon=lon, area_meters=area_meters,
            )
            result["primary_image"] = tile_result["image"]
            result["metadata"]["sources_used"].append("Esri World Imagery (fallback)")
            result["metadata"]["primary_path"] = tile_result["image_path"]
            logger.info("[OK] Esri World Imagery fallback succeeded")
        except Exception as e:
            logger.error(f"Esri fallback also failed: {e}")

        return result


# ─────────────────────────────────────────────
# Module-level convenience function
# ─────────────────────────────────────────────

_router_instance: Optional[ImageryRouter] = None


def get_imagery_router() -> ImageryRouter:
    """Get or create the singleton ImageryRouter instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = ImageryRouter()
    return _router_instance


def fetch_disaster_imagery(
    lat: float,
    lon: float,
    disaster_type: str = "flood",
    area_meters: float = 2000.0,
    lookback_days: int = 14,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function: fetch all disaster imagery for a location.
    Uses the singleton ImageryRouter.
    """
    router = get_imagery_router()
    return router.fetch_disaster_imagery(
        lat=lat,
        lon=lon,
        disaster_type=disaster_type,
        area_meters=area_meters,
        lookback_days=lookback_days,
        **kwargs,
    )
