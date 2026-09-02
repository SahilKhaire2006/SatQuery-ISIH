"""
Inference API for Text-Guided Grounding Specialist Model

Provides the primary stable interface `ground(image, query)` for executing text-guided
visual grounding on satellite imagery.
"""

from typing import Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image

from .model import TextGuidedGroundingModel

# Global model instance for lightweight reuse across inference calls
_GROUNDING_MODEL_INSTANCE: Optional[TextGuidedGroundingModel] = None


def get_grounding_model(model_name: str = "google/owlvit-base-patch32") -> TextGuidedGroundingModel:
    """Retrieve or initialize singleton model instance."""
    global _GROUNDING_MODEL_INSTANCE
    if _GROUNDING_MODEL_INSTANCE is None:
        _GROUNDING_MODEL_INSTANCE = TextGuidedGroundingModel(model_name=model_name)
    return _GROUNDING_MODEL_INSTANCE


def ground(
    image: Union[np.ndarray, Image.Image, str, Path],
    query: str,
    model_name: str = "google/owlvit-base-patch32",
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Perform text-guided visual grounding for a given satellite image and referring expression.

    Args:
        image: Input satellite image. Can be:
               - numpy array (H, W, 3)
               - PIL Image object
               - file path string or Path object
        query: Natural language referring expression string (e.g., "a round water reservoir near the road")
        model_name: Neural vision-language model identifier
        top_k: Number of candidate bounding boxes to return

    Returns:
        Dict containing:
            - "bbox": [xmin, ymin, xmax, ymax] floating-point coordinates
            - "confidence": float score (0.0 to 1.0)
            - "label": query phrase
            - "candidates": list of top-k candidate bounding boxes and confidence scores
            - "image_dimensions": [width, height]

    Contract Guarantee:
        - No dataset-specific hardcoded coordinates or fixed region lookup tables are used.
        - Output is derived directly from neural model inference.
    """
    if isinstance(image, (str, Path)):
        img_path = Path(image)
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        image_input = Image.open(img_path).convert("RGB")
    else:
        image_input = image

    model = get_grounding_model(model_name=model_name)
    return model.predict(image=image_input, query=query, top_k=top_k)
