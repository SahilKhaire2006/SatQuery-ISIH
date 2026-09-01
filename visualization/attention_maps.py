import cv2
import numpy as np
import base64
from typing import Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AttentionMapGenerator:
    """
    Generates GradCAM-style spatial attention maps and heatmaps over satellite imagery.
    """

    def generate_attention_heatmap(
        self,
        image: np.ndarray,
        attention_mask: np.ndarray = None,
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, str]:
        """
        Generate colorized GradCAM heatmap overlay on input image.
        Returns (blended_image_np, base64_png_string).
        """
        try:
            if image is None or image.size == 0:
                image = np.zeros((200, 200, 3), dtype=np.uint8)

            img_copy = image.copy()
            if img_copy.dtype != np.uint8:
                if img_copy.max() <= 1.0:
                    img_copy = (img_copy * 255).astype(np.uint8)
                else:
                    img_copy = img_copy.astype(np.uint8)

            if img_copy.ndim == 2:
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
            elif img_copy.ndim == 3 and img_copy.shape[2] == 4:
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGBA2BGR)
            elif img_copy.ndim == 3 and img_copy.shape[2] == 1:
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)

            h, w = img_copy.shape[:2]

            if attention_mask is None:
                # Generate synthetic Gaussian spatial attention center if not provided
                y, x = np.ogrid[:h, :w]
                center_y, center_x = h / 2.0, w / 2.0
                sigma = max(1.0, min(h, w) / 4.0)
                attention_mask = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

            # Normalize mask 0-255
            norm_mask = cv2.normalize(attention_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if norm_mask.shape[:2] != (h, w):
                norm_mask = cv2.resize(norm_mask, (w, h))

            # Apply JET colormap
            heatmap = cv2.applyColorMap(norm_mask, cv2.COLORMAP_JET)

            if heatmap.shape != img_copy.shape:
                heatmap = cv2.resize(heatmap, (w, h))

            # Blend image and heatmap
            blended = cv2.addWeighted(img_copy, 1.0 - alpha, heatmap, alpha, 0)

            # Encode to Base64
            _, buffer = cv2.imencode('.png', blended)
            b64_str = base64.b64encode(buffer).decode('utf-8')

            return blended, b64_str

        except Exception as e:
            logger.error(f"Error in AttentionMapGenerator: {e}")
            fallback_img = np.zeros((200, 200, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.png', fallback_img)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            return fallback_img, b64_str
