import numpy as np
from typing import List, Dict, Any, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)


class TileProcessor:
    """
    Decomposes large satellite images into sliding-window tiles for high-resolution inference and merges tiled detections.
    """

    def __init__(self, tile_size: int = 256, overlap: int = 32):
        self.tile_size = tile_size
        self.overlap = overlap

    def generate_tiles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Decompose input satellite image into tiles with stride = (tile_size - overlap).
        Returns list of tile dicts: {'tile': np_array, 'x_offset': int, 'y_offset': int, 'tile_id': str}.
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        stride = max(16, self.tile_size - self.overlap)
        tiles = []

        tile_index = 0
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x_end = min(w, x + self.tile_size)
                y_end = min(h, y + self.tile_size)

                tile_crop = image[y:y_end, x:x_end]
                tiles.append({
                    'tile_id': f"tile_{tile_index}",
                    'tile': tile_crop,
                    'x_offset': x,
                    'y_offset': y,
                    'width': x_end - x,
                    'height': y_end - y
                })
                tile_index += 1

        logger.info(f"Generated {len(tiles)} tiles ({self.tile_size}x{self.tile_size}) for image of shape ({h}, {w})")
        return tiles

    def merge_tiled_detections(
        self,
        tiled_detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remap tile-local bounding boxes back to global image coordinates.
        """
        global_detections = []
        for item in tiled_detections:
            x_off = item.get('x_offset', 0)
            y_off = item.get('y_offset', 0)
            det = item.get('detection', {})

            if 'bbox' in det:
                lx1, ly1, lx2, ly2 = det['bbox']
                gx1 = lx1 + x_off
                gy1 = ly1 + y_off
                gx2 = lx2 + x_off
                gy2 = ly2 + y_off

                g_det = dict(det)
                g_det['bbox'] = [gx1, gy1, gx2, gy2]
                global_detections.append(g_det)

        return global_detections
