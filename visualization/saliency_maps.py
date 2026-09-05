import cv2
import numpy as np
import base64
from typing import Tuple, List, Dict, Any, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SaliencyMapGenerator:
    """
    Computes query-aware spatial saliency maps highlighting key regions
    in satellite imagery based on spectral, structural, and detection features.
    """

    def compute_saliency_map(
        self,
        image: np.ndarray,
        detections: Optional[List[Dict[str, Any]]] = None,
        query: str = ""
    ) -> Tuple[np.ndarray, str]:
        """
        Compute a multi-channel spatial saliency map highlighting regions
        most relevant to the query and model analysis.

        Args:
            image: Input image (RGB numpy array)
            detections: Optional detection results for query-aware saliency
            query: Query string for context-aware highlighting

        Returns:
            (saliency_color_np, base64_png_string)
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

            # Convert RGB to BGR for OpenCV
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
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

            # --- Multi-channel saliency ---

            # 1. Edge/structure saliency (Sobel gradient magnitude)
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_mag = cv2.magnitude(gx, gy)
            edge_saliency = cv2.normalize(edge_mag, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

            # 2. Color contrast saliency (deviation from mean color)
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            mean_lab = lab.mean(axis=(0, 1))
            color_dist = np.sqrt(np.sum((lab - mean_lab) ** 2, axis=2))
            color_saliency = cv2.normalize(color_dist, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

            # 3. Local contrast saliency (center-surround difference)
            blur_small = cv2.GaussianBlur(gray, (5, 5), 0)
            blur_large = cv2.GaussianBlur(gray, (25, 25), 0)
            center_surround = np.abs(blur_small - blur_large)
            cs_saliency = cv2.normalize(center_surround, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

            # 4. Query-aware boost from detections
            detection_boost = np.zeros((h, w), dtype=np.float32)
            if detections and len(detections) > 0:
                for det in detections:
                    bbox = det.get('bbox', [0, 0, 0, 0])
                    conf = det.get('confidence', 0.5)
                    xmin, ymin, xmax, ymax = [max(0, int(v)) for v in bbox]
                    xmax = min(xmax, w)
                    ymax = min(ymax, h)
                    if xmax > xmin and ymax > ymin:
                        detection_boost[ymin:ymax, xmin:xmax] += conf
                if detection_boost.max() > 0:
                    detection_boost = cv2.normalize(detection_boost, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
                    # Smooth the detection boost
                    detection_boost = cv2.GaussianBlur(detection_boost, (0, 0), sigmaX=max(5, min(h, w) / 30.0))
                    detection_boost = cv2.normalize(detection_boost, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

            # Combine saliency channels with weights
            if detections and len(detections) > 0:
                combined = 0.25 * edge_saliency + 0.2 * color_saliency + 0.15 * cs_saliency + 0.4 * detection_boost
            else:
                combined = 0.4 * edge_saliency + 0.35 * color_saliency + 0.25 * cs_saliency

            # Smooth final saliency
            combined = cv2.GaussianBlur(combined, (0, 0), sigmaX=max(2, min(h, w) / 60.0))
            saliency_norm = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Apply INFERNO colormap for saliency (more perceptually uniform than VIRIDIS)
            saliency_color = cv2.applyColorMap(saliency_norm, cv2.COLORMAP_INFERNO)

            # Blend with original image for context
            blended = cv2.addWeighted(img_bgr, 0.35, saliency_color, 0.65, 0)

            # Add label
            cv2.putText(blended, "Spatial Saliency Map", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(blended, "Spatial Saliency Map", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            # Add legend bar
            legend_h = 25
            legend_bar = np.zeros((legend_h, w, 3), dtype=np.uint8)
            gradient = np.linspace(0, 255, w).astype(np.uint8)
            gradient_2d = np.tile(gradient, (legend_h, 1))
            legend_bar = cv2.applyColorMap(gradient_2d, cv2.COLORMAP_INFERNO)
            cv2.putText(legend_bar, "Low", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(legend_bar, "High Saliency", (w - 130, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            blended = np.vstack([blended, legend_bar])

            _, buffer = cv2.imencode('.png', blended)
            b64_str = base64.b64encode(buffer).decode('utf-8')

            return blended, b64_str

        except Exception as e:
            logger.error(f"Error in SaliencyMapGenerator: {e}", exc_info=True)
            fallback_img = np.zeros((200, 200, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.png', fallback_img)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            return fallback_img, b64_str
