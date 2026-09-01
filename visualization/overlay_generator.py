import cv2
import numpy as np
import base64
from typing import List, Dict, Any, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)


class OverlayGenerator:
    """
    Renders visual bounding box overlays, label annotations, and confidence badges onto satellite imagery.
    """

    def __init__(self):
        self.colors = [
            (0, 255, 0),    # Bright Green
            (255, 0, 0),    # Bright Blue
            (0, 165, 255),  # Orange
            (255, 0, 255),  # Magenta
            (0, 255, 255)   # Yellow
        ]

    def draw_bounding_boxes(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, str]:
        """
        Draw bounding boxes and confidence labels onto image array.
        Returns (annotated_image_np, base64_png_string).
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

            for idx, det in enumerate(detections):
                bbox = det.get('bbox', [0, 0, w, h])
                label = det.get('label', 'target')
                conf = det.get('confidence', 0.85)

                xmin, ymin, xmax, ymax = [int(v) for v in bbox]
                color = self.colors[idx % len(self.colors)]

                # Draw rectangle box
                cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), color, 2)

                # Draw label banner
                text = f"{label} {conf:.2f}"
                (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    img_copy,
                    (xmin, max(0, ymin - text_h - 4)),
                    (xmin + text_w, ymin),
                    color,
                    -1
                )
                cv2.putText(
                    img_copy,
                    text,
                    (xmin, max(text_h, ymin - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )

            # Encode to Base64 PNG
            _, buffer = cv2.imencode('.png', img_copy)
            b64_str = base64.b64encode(buffer).decode('utf-8')

            return img_copy, b64_str

        except Exception as e:
            logger.error(f"Error in OverlayGenerator: {e}")
            fallback_img = np.zeros((200, 200, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.png', fallback_img)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            return fallback_img, b64_str
