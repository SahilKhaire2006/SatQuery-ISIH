import cv2
import numpy as np
import base64
from typing import Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SaliencyMapGenerator:
    """
    Computes spatial activation saliency maps highlighting key land-cover regions.
    """

    def compute_saliency_map(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Compute visual saliency map highlighting high-contrast spatial regions.
        """
        try:
            if image is None or image.size == 0:
                image = np.zeros((200, 200, 3), dtype=np.uint8)

            img_copy = image.copy()
            if img_copy.ndim == 2:
                gray = img_copy
            elif img_copy.ndim == 3 and img_copy.shape[2] == 4:
                gray = cv2.cvtColor(img_copy, cv2.COLOR_RGBA2GRAY)
            elif img_copy.ndim == 3 and img_copy.shape[2] == 1:
                gray = img_copy[:, :, 0]
            else:
                gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

            # Compute gradient magnitude as saliency metric
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)

            saliency_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            saliency_color = cv2.applyColorMap(saliency_norm, cv2.COLORMAP_VIRIDIS)

            _, buffer = cv2.imencode('.png', saliency_color)
            b64_str = base64.b64encode(buffer).decode('utf-8')

            return saliency_color, b64_str

        except Exception as e:
            logger.error(f"Error in SaliencyMapGenerator: {e}")
            fallback_img = np.zeros((200, 200, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.png', fallback_img)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            return fallback_img, b64_str
