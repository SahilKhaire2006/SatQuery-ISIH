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

            for idx, det in enumerate(detections, start=1):  # Start counting from 1
                bbox = det.get('bbox', [0, 0, w, h])
                label = det.get('label', 'target')
                conf = det.get('confidence', 0.85)

                xmin, ymin, xmax, ymax = [int(v) for v in bbox]
                color = self.colors[idx % len(self.colors)]

                # Draw rectangle box with thicker line for visibility
                cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), color, 3)

                # Draw numbered label (like reference image: "1", "2", etc.)
                label_text = str(idx)
                font_scale = 0.8
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Draw label background circle/box
                label_size = max(text_w, text_h) + 8
                label_x = xmin + 5
                label_y = ymin + 5
                cv2.rectangle(
                    img_copy,
                    (label_x, label_y),
                    (label_x + label_size, label_y + label_size),
                    color,
                    -1
                )
                
                # Draw number
                text_x = label_x + (label_size - text_w) // 2
                text_y = label_y + (label_size + text_h) // 2
                cv2.putText(
                    img_copy,
                    label_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),  # White text
                    thickness,
                    cv2.LINE_AA
                )
            
            # Add total count overlay (bottom-left corner like reference)
            if detections:
                count_text = f"Total Buildings: {len(detections)}"
                font_scale = 0.7
                thickness = 2
                (count_w, count_h), baseline = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Dark background for count
                cv2.rectangle(
                    img_copy,
                    (10, h - count_h - 25),
                    (count_w + 30, h - 10),
                    (0, 0, 0),
                    -1
                )
                cv2.rectangle(
                    img_copy,
                    (10, h - count_h - 25),
                    (count_w + 30, h - 10),
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    img_copy,
                    count_text,
                    (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
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
