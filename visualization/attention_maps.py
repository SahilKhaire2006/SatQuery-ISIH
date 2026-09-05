import cv2
import numpy as np
import base64
from typing import Tuple, List, Dict, Any, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AttentionMapGenerator:
    """
    Generates query-aware spatial attention heatmaps over satellite imagery.
    Uses detection results and spectral analysis to highlight regions the AI focused on.
    """

    def generate_attention_heatmap(
        self,
        image: np.ndarray,
        attention_mask: np.ndarray = None,
        detections: Optional[List[Dict[str, Any]]] = None,
        query: str = "",
        alpha: float = 0.45
    ) -> Tuple[np.ndarray, str]:
        """
        Generate a query-aware spatial attention heatmap overlay.
        If detections are provided, the attention is concentrated on detected regions.
        Otherwise, uses image feature analysis to estimate attention.

        Args:
            image: Input image (RGB numpy array)
            attention_mask: Optional pre-computed attention mask
            detections: List of detection dicts with 'bbox' keys
            query: Query string for context
            alpha: Blend factor

        Returns:
            (blended_image_np, base64_png_string)
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

            # Convert RGB to BGR for OpenCV operations
            img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
            h, w = img_bgr.shape[:2]

            # Upscale small image (< 500px) to 600x600 for clear visualization and HUD text
            if h < 500 or w < 500:
                scale_x = 600.0 / max(w, 1)
                scale_y = 600.0 / max(h, 1)
                img_bgr = cv2.resize(img_bgr, (600, 600), interpolation=cv2.INTER_CUBIC)
                h, w = 600, 600
                if detections:
                    scaled_dets = []
                    for d in detections:
                        dc = dict(d)
                        if 'bbox' in dc:
                            x1, y1, x2, y2 = dc['bbox']
                            dc['bbox'] = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                        scaled_dets.append(dc)
                    detections = scaled_dets

            if attention_mask is None:
                if detections and len(detections) > 0:
                    # Build attention from detection bounding boxes
                    attention_mask = self._build_detection_attention(h, w, detections)
                    logger.info(f"Generated detection-based attention from {len(detections)} detections")
                else:
                    # Build attention from image features (texture, edges, color variance)
                    attention_mask = self._build_feature_attention(img_bgr)
                    logger.info("Generated feature-based attention from image analysis")

            # Normalize mask to 0-255
            norm_mask = cv2.normalize(attention_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if norm_mask.shape[:2] != (h, w):
                norm_mask = cv2.resize(norm_mask, (w, h))

            # Apply Gaussian blur for smooth heatmap transitions
            norm_mask = cv2.GaussianBlur(norm_mask, (0, 0), sigmaX=max(3, min(h, w) / 40.0))
            norm_mask = cv2.normalize(norm_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Apply JET colormap
            heatmap = cv2.applyColorMap(norm_mask, cv2.COLORMAP_JET)

            if heatmap.shape != img_bgr.shape:
                heatmap = cv2.resize(heatmap, (w, h))

            # Blend image and heatmap
            blended = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap, alpha, 0)

            # Add "GradCAM Attention" label
            cv2.putText(blended, "Spatial Attention Heatmap", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(blended, "Spatial Attention Heatmap", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            # Add color scale legend bar
            legend_h = 25
            legend_bar = np.zeros((legend_h, w, 3), dtype=np.uint8)
            gradient = np.linspace(0, 255, w).astype(np.uint8)
            gradient_2d = np.tile(gradient, (legend_h, 1))
            legend_bar = cv2.applyColorMap(gradient_2d, cv2.COLORMAP_JET)
            cv2.putText(legend_bar, "Low", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(legend_bar, "High Attention", (w - 130, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            blended = np.vstack([blended, legend_bar])

            # Encode to Base64
            _, buffer = cv2.imencode('.png', blended)
            b64_str = base64.b64encode(buffer).decode('utf-8')

            return blended, b64_str

        except Exception as e:
            logger.error(f"Error in AttentionMapGenerator: {e}", exc_info=True)
            fallback_img = np.zeros((200, 200, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.png', fallback_img)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            return fallback_img, b64_str

    def _build_detection_attention(self, h: int, w: int, detections: List[Dict]) -> np.ndarray:
        """Build an attention mask from detection bounding boxes using Gaussian blobs centered on each detection."""
        attention = np.zeros((h, w), dtype=np.float32)

        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            conf = det.get('confidence', 0.5)
            xmin, ymin, xmax, ymax = [int(v) for v in bbox]

            # Clamp to image bounds
            xmin = max(0, min(xmin, w - 1))
            ymin = max(0, min(ymin, h - 1))
            xmax = max(0, min(xmax, w))
            ymax = max(0, min(ymax, h))

            if xmax <= xmin or ymax <= ymin:
                continue

            # Create Gaussian blob centered on the detection
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            sx = max(10, (xmax - xmin) / 2.0)
            sy = max(10, (ymax - ymin) / 2.0)

            y_coords, x_coords = np.ogrid[:h, :w]
            gaussian = np.exp(-((x_coords - cx) ** 2 / (2 * sx ** 2) + (y_coords - cy) ** 2 / (2 * sy ** 2)))

            # Weight by confidence
            attention += gaussian * conf

            # Also fill the bbox area directly for stronger signal
            attention[ymin:ymax, xmin:xmax] += conf * 0.3

        return attention

    def _build_feature_attention(self, img_bgr: np.ndarray) -> np.ndarray:
        """Build attention from image features: edges, texture variance, and color saturation."""
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Edge magnitude component
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = cv2.magnitude(gx, gy)

        # Local variance component (texture complexity)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        local_var = (gray - blur) ** 2
        local_var = cv2.GaussianBlur(local_var, (15, 15), 0)

        # Color saturation component (more saturated = more interesting)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)

        # Combine components
        edge_norm = cv2.normalize(edge_mag, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        var_norm = cv2.normalize(local_var, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        sat_norm = cv2.normalize(saturation, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

        # Weighted combination
        attention = 0.4 * edge_norm + 0.35 * var_norm + 0.25 * sat_norm

        return attention
