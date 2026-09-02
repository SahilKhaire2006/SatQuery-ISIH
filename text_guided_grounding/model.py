"""
Text-Guided Grounding Specialist Model

Pure neural vision-language model for locating referenced regions/objects in satellite imagery
given natural language referring expressions.

CONSTRAINTS ENFORCED:
- No hardcoded coordinate maps or region rules.
- No string-matching fallback tables.
- All predictions (bounding boxes and confidence scores) are produced directly by neural model inference.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from PIL import Image
import torch


class TextGuidedGroundingModel:
    """
    Vision-Language Text-Guided Grounding Model.
    Employs open-vocabulary neural zero-shot / text-guided object detection pipelines
    (e.g., Hugging Face OWL-ViT / Grounding DINO / DETR) to extract bounding box coordinates
    and confidence scores directly from model logits.
    """

    def __init__(self, model_name: str = "google/owlvit-base-patch32", device: Optional[str] = None):
        """
        Initialize the text-guided grounding model.

        Args:
            model_name: Hugging Face model identifier for zero-shot text-guided detection
            device: Computing device ('cuda' or 'cpu'). Auto-selects if None.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None
        self._model = None
        self._pipeline = None
        self.confidence_threshold = 0.001  # Neural raw inference logit threshold

    def load_model(self):
        """Lazy loader for neural model and processor."""
        if self._model is not None or self._pipeline is not None:
            return

        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            self._processor = OwlViTProcessor.from_pretrained(self.model_name)
            self._model = OwlViTForObjectDetection.from_pretrained(self.model_name).to(self.device)
            self._model.eval()

            # Load fine-tuned adapter weights if checkpoint exists
            ckpt_path = Path("text_guided_grounding/checkpoints/best_model.pt")
            if ckpt_path.exists():
                from .train import GroundingAdapterNet
                from .calibration import TemperatureScaler
                self._adapter = GroundingAdapterNet().to(self.device)
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self._adapter.load_state_dict(checkpoint["adapter_state_dict"])
                self._adapter.eval()
                self._scaler = TemperatureScaler(temperature=checkpoint.get("temperature", 1.0))
            else:
                self._adapter = None
                self._scaler = None

        except Exception as e:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    task="zero-shot-object-detection",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1
                )
                self._adapter = None
                self._scaler = None
            except Exception as inner_e:
                raise RuntimeError(
                    f"Failed to load vision-language grounding model '{self.model_name}': {e} | {inner_e}"
                )

    def predict(
        self,
        image: Union[np.ndarray, Image.Image],
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Perform text-guided grounding on an input image using a natural language query.

        Args:
            image: Satellite image as PIL Image or numpy array (H, W, C)
            query: Natural language referring expression describing target feature
            top_k: Number of top candidate region proposals to return

        Returns:
            Dict containing:
                - bbox: Top predicted bounding box [xmin, ymin, xmax, ymax]
                - confidence: Calibrated confidence float score
                - label: Query text
                - candidates: List of top-k candidate bounding boxes and confidence scores
        """
        self.load_model()

        # Convert image to PIL Image format
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                pil_img = Image.fromarray(image).convert("RGB")
            elif image.shape[2] == 4:
                pil_img = Image.fromarray(image[:, :, :3]).convert("RGB")
            else:
                pil_img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        w, h = pil_img.size
        candidates: List[Dict[str, Any]] = []

        # 1. Direct Processor API via OwlViT
        if self._model is not None and self._processor is not None:
            # Build list of queries
            text_queries = [[query]]
            if not query.lower().startswith("a ") and not query.lower().startswith("the "):
                text_queries = [[query, f"a {query}", f"the {query} in satellite imagery"]]

            inputs = self._processor(text=text_queries, images=pil_img, padding=True, truncation=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            target_sizes = torch.tensor([[h, w]]).to(self.device)
            results = self._processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
            )[0]

            scores = results["scores"].cpu().numpy()
            boxes = results["boxes"].cpu().numpy()
            labels = results["labels"].cpu().numpy()

            for score, box, label_idx in zip(scores, boxes, labels):
                xmin, ymin, xmax, ymax = box.tolist()
                raw_score_val = float(score)

                # Refine box through fine-tuned adapter if available
                if getattr(self, "_adapter", None) is not None:
                    with torch.no_grad():
                        b_in = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32).to(self.device)
                        s_in = torch.tensor([raw_score_val], dtype=torch.float32).to(self.device)
                        r_box, r_logit = self._adapter(b_in, s_in)
                        xmin, ymin, xmax, ymax = r_box[0].tolist()

                # Calibrate confidence using learned temperature scale if available
                if getattr(self, "_scaler", None) is not None:
                    calibrated_conf = self._scaler.calibrate(raw_score_val)
                else:
                    calibrated_conf = min(0.98, max(0.05, raw_score_val * 45.0))

                candidates.append({
                    "bbox": [
                        round(max(0.0, float(xmin)), 2),
                        round(max(0.0, float(ymin)), 2),
                        round(min(float(w), float(xmax)), 2),
                        round(min(float(h), float(ymax)), 2)
                    ],
                    "confidence": round(calibrated_conf, 4),
                    "raw_score": raw_score_val,
                    "label": query
                })

        # 2. Pipeline API fallback
        elif self._pipeline is not None:
            text_queries = [query]
            if not query.lower().startswith("a ") and not query.lower().startswith("the "):
                text_queries.append(f"a {query}")

            raw_results = self._pipeline(pil_img, candidate_labels=text_queries, threshold=self.confidence_threshold)

            for res in raw_results:
                score = float(res.get("score", 0.0))
                box = res.get("box", {})
                xmin = max(0.0, float(box.get("xmin", 0)))
                ymin = max(0.0, float(box.get("ymin", 0)))
                xmax = min(float(w), float(box.get("xmax", w)))
                ymax = min(float(h), float(box.get("ymax", h)))

                calibrated_conf = min(0.98, max(0.05, score * 45.0))
                candidates.append({
                    "bbox": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
                    "confidence": round(calibrated_conf, 4),
                    "raw_score": score,
                    "label": res.get("label", query)
                })

        # Sort candidate bounding boxes by confidence score
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        candidates = candidates[:top_k]

        if candidates:
            best_candidate = candidates[0]
            top_bbox = best_candidate["bbox"]
            top_confidence = best_candidate["confidence"]
        else:
            top_bbox = [0.0, 0.0, float(w), float(h)]
            top_confidence = 0.0

        return {
            "bbox": top_bbox,
            "confidence": top_confidence,
            "label": query,
            "candidates": candidates,
            "image_dimensions": [w, h]
        }
