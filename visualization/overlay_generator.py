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
        Draw exact pixel segmentation overlays, contour outlines, and text tags onto satellite image array.
        Highlights exact features:
          - Water Bodies: Semi-transparent Blue/Cyan pixel overlay + Cyan outline
          - Buildings: Semi-transparent Violet/Purple pixel overlay + Violet outline
          - Vegetation: Semi-transparent Emerald Green pixel overlay + Green outline
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
                img_copy = np.stack([img_copy] * 3, axis=-1)
            elif img_copy.ndim == 3 and img_copy.shape[2] == 4:
                img_copy = img_copy[:, :, :3]
            elif img_copy.ndim == 3 and img_copy.shape[2] == 1:
                img_copy = np.concatenate([img_copy] * 3, axis=-1)

            # Convert RGB to BGR for OpenCV drawing operations
            img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
            h, w, _ = img_bgr.shape

            # Compute spectral pixel masks on the image array
            r = img_copy[:, :, 0].astype(np.float32)
            g = img_copy[:, :, 1].astype(np.float32)
            b = img_copy[:, :, 2].astype(np.float32)
            eps = 1e-5

            ndwi = (b + g - 2.0 * r) / (b + g + 2.0 * r + eps)
            ndvi = (g - r) / (g + r + eps)
            exg = (2.0 * g - r - b) / 255.0

            # 1. Exact Water Mask (Universal Optical Water Rule)
            c1 = (b / (r + eps) >= 1.20) & (b / (g + eps) >= 1.04) & (r <= 115) & (b >= 25)
            c2 = (ndwi > 0.06) & (b > r * 1.12) & (r < 115)
            water_mask = (c1 | c2) & (ndvi <= 0.08) & ~((r > 200) & (g > 200) & (b > 200))
            water_uint8 = (water_mask.astype(np.uint8) * 255)
            w_closed = cv2.morphologyEx(water_uint8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)))
            w_closed[:15, :] = 0; w_closed[-15:, :] = 0; w_closed[:, :15] = 0; w_closed[:, -15:] = 0
            water_cleaned = cv2.morphologyEx(w_closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

            # 2. Exact Vegetation Mask
            veg_mask = (ndvi > 0.06) & (g > r * 1.08) & (exg > 0.015) & (water_cleaned == 0) & ~((r > 200) & (g > 200) & (b > 200))
            veg_uint8 = (veg_mask.astype(np.uint8) * 255)
            veg_cleaned = cv2.morphologyEx(veg_uint8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

            # Categorize detection intent
            labels_str = " ".join([str(d.get('label', '')).lower() for d in detections])
            has_water = 'water' in labels_str
            has_veg = 'veg' in labels_str or 'tree' in labels_str
            has_bldg = 'build' in labels_str or 'structure' in labels_str

            overlay = img_bgr.copy()

            # A. Apply Water Blue Highlight (BGR: 255, 195, 0)
            if (has_water or not (has_veg or has_bldg)) and np.sum(water_cleaned > 0) > 0:
                water_color = np.array([255, 195, 0], dtype=np.uint8)
                overlay[water_cleaned > 0] = cv2.addWeighted(
                    img_bgr[water_cleaned > 0], 0.55,
                    np.tile(water_color, (np.sum(water_cleaned > 0), 1)), 0.45, 0
                )

            # B. Apply Vegetation Green Highlight (BGR: 0, 230, 115)
            if (has_veg or not (has_water or has_bldg)) and np.sum(veg_cleaned > 0) > 0:
                veg_color = np.array([0, 230, 115], dtype=np.uint8)
                overlay[veg_cleaned > 0] = cv2.addWeighted(
                    img_bgr[veg_cleaned > 0], 0.60,
                    np.tile(veg_color, (np.sum(veg_cleaned > 0), 1)), 0.40, 0
                )

            # C. Apply Building Violet Highlight (BGR: 245, 92, 218)
            bldg_mask = np.zeros((h, w), dtype=np.uint8)
            for det in detections:
                label = str(det.get('label', '')).lower()
                if 'build' in label or 'structure' in label:
                    bbox = det.get('bbox', [0, 0, 0, 0])
                    xmin, ymin, xmax, ymax = [max(0, int(v)) for v in bbox]
                    if xmax > xmin and ymax > ymin:
                        bldg_mask[ymin:ymax, xmin:xmax] = 255

            # Mask out water and heavy vegetation from building overlay
            bldg_mask[water_cleaned > 0] = 0
            bldg_mask[veg_cleaned > 0] = 0

            if (has_bldg or not (has_water or has_veg)) and np.sum(bldg_mask > 0) > 0:
                bldg_color = np.array([245, 92, 218], dtype=np.uint8)
                overlay[bldg_mask > 0] = cv2.addWeighted(
                    img_bgr[bldg_mask > 0], 0.55,
                    np.tile(bldg_color, (np.sum(bldg_mask > 0), 1)), 0.45, 0
                )

            # Draw Water body contours & text tags
            if (has_water or not (has_veg or has_bldg)):
                w_contours, _ = cv2.findContours(water_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                water_idx = 0
                for cnt in w_contours:
                    if cv2.contourArea(cnt) >= 2500:
                        water_idx += 1
                        cv2.drawContours(overlay, [cnt], -1, (255, 255, 0), 2, lineType=cv2.LINE_AA)
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        tag = f"Water #{water_idx}"
                        font_scale = 0.6
                        thickness = 1
                        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        lx, ly = x, max(y - th - 6, 0)
                        cv2.rectangle(overlay, (lx, ly), (lx + tw + 6, ly + th + 6), (255, 255, 0), -1)
                        cv2.putText(overlay, tag, (lx + 3, ly + th + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            # Draw Vegetation contours & text tags
            if (has_veg or not (has_water or has_bldg)):
                v_contours, _ = cv2.findContours(veg_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                veg_idx = 0
                for cnt in v_contours:
                    if cv2.contourArea(cnt) >= 1500:
                        veg_idx += 1
                        cv2.drawContours(overlay, [cnt], -1, (0, 230, 115), 2, lineType=cv2.LINE_AA)
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        tag = f"Veg #{veg_idx}"
                        font_scale = 0.6
                        thickness = 1
                        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        lx, ly = x, max(y - th - 6, 0)
                        cv2.rectangle(overlay, (lx, ly), (lx + tw + 6, ly + th + 6), (0, 230, 115), -1)
                        cv2.putText(overlay, tag, (lx + 3, ly + th + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            # Draw Building outlines & text tags
            if has_bldg:
                bldg_idx = 0
                for det in detections:
                    label = str(det.get('label', '')).lower()
                    if 'build' in label or 'structure' in label:
                        bldg_idx += 1
                        bbox = det.get('bbox', [0, 0, 0, 0])
                        xmin, ymin, xmax, ymax = [max(0, int(v)) for v in bbox]
                        color = (245, 92, 218)
                        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, 2)
                        tag = f"Building #{bldg_idx}"
                        font_scale = 0.6
                        thickness = 1
                        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        lx, ly = xmin, max(ymin - th - 6, 0)
                        cv2.rectangle(overlay, (lx, ly), (lx + tw + 6, ly + th + 6), color, -1)
                        cv2.putText(overlay, tag, (lx + 3, ly + th + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Add total count overlay (bottom-left corner)
            if detections:
                if 'water' in labels_str and 'build' not in labels_str and 'veg' not in labels_str:
                    count_text = f"Total Water Bodies: {len(detections)}"
                elif ('veg' in labels_str or 'tree' in labels_str) and 'build' not in labels_str and 'water' not in labels_str:
                    count_text = f"Total Vegetation Regions: {len(detections)}"
                elif 'build' in labels_str or 'structure' in labels_str:
                    count_text = f"Total Buildings: {len(detections)}"
                else:
                    count_text = f"Total Objects: {len(detections)}"

                font_scale = 0.7
                thickness = 2
                (count_w, count_h), baseline = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(overlay, (10, h - count_h - 25), (count_w + 30, h - 10), (0, 0, 0), -1)
                cv2.rectangle(overlay, (10, h - count_h - 25), (count_w + 30, h - 10), (255, 255, 255), 2)
                cv2.putText(overlay, count_text, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Encode to Base64 PNG
            _, buffer = cv2.imencode('.png', overlay)
            b64_str = base64.b64encode(buffer).decode('utf-8')

            return overlay, b64_str

        except Exception as e:
            logger.error(f"Error in OverlayGenerator: {e}", exc_info=True)
            fallback_img = np.zeros((200, 200, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.png', fallback_img)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            return fallback_img, b64_str
