"""
Visualizer for Text-Guided Grounding Predictions

Renders bounding box overlays on satellite images, comparing ground-truth boxes
with model-predicted boxes and confidence scores.
"""

from typing import Union, List, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def visualize_grounding(
    image: Union[np.ndarray, Image.Image, str, Path],
    pred_bbox: List[float],
    confidence: float,
    gt_bbox: Optional[List[float]] = None,
    query: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Overlay predicted bounding box and optional ground-truth box on satellite image.

    Args:
        image: Satellite image input
        pred_bbox: Predicted box [xmin, ymin, xmax, ymax]
        confidence: Model confidence score float
        gt_bbox: Optional ground-truth box [xmin, ymin, xmax, ymax]
        query: Referring expression query string
        output_path: Optional file path to save rendered overlay image

    Returns:
        Rendered image as RGB numpy array (H, W, 3)
    """
    if isinstance(image, (str, Path)):
        img_arr = cv2.imread(str(image))
        if img_arr is not None:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Could not read image from {image}")
    elif isinstance(image, Image.Image):
        img_arr = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        img_arr = image.copy()
        if img_arr.ndim == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    h, w, _ = img_arr.shape
    rendered = img_arr.copy()

    # Draw Ground Truth BBox if provided (Lime green: (0, 255, 0))
    if gt_bbox and len(gt_bbox) == 4:
        gx1, gy1, gx2, gy2 = [int(v) for v in gt_bbox]
        cv2.rectangle(rendered, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        cv2.putText(
            rendered,
            "GT",
            (gx1, max(15, gy1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    # Draw Predicted BBox (Cyan/Electric Blue: (0, 215, 255))
    if pred_bbox and len(pred_bbox) == 4:
        px1, py1, px2, py2 = [int(v) for v in pred_bbox]
        cv2.rectangle(rendered, (px1, py1), (px2, py2), (255, 215, 0), 3)

        label_str = f"Pred: {confidence:.2%}"
        if query:
            label_str = f"'{query[:25]}' | {confidence:.2%}"

        # Draw text background banner for high readability
        (tw, th), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_bg_y1 = max(0, py1 - th - 8)
        cv2.rectangle(rendered, (px1, text_bg_y1), (px1 + tw + 6, py1), (255, 215, 0), cv2.FILLED)
        cv2.putText(
            rendered,
            label_str,
            (px1 + 3, py1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Convert RGB to BGR for OpenCV imwrite
        cv2.imwrite(str(out_p), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))

    return rendered
