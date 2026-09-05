import os
import cv2
import base64
import numpy as np
from typing import Dict, Any, List, Optional
from visualization.overlay_generator import OverlayGenerator
from visualization.attention_maps import AttentionMapGenerator
from visualization.saliency_maps import SaliencyMapGenerator
from utils.logger import setup_logger

logger = setup_logger(__name__)


class EvidenceCompiler:
    """
    Compiles multi-source visual and textual evidence into consolidated evidence packages (USP-2).
    """

    def __init__(self):
        self.overlay_generator = OverlayGenerator()
        self.attention_generator = AttentionMapGenerator()
        self.saliency_generator = SaliencyMapGenerator()

    def _extract_detections(self, obj: Any) -> List[Dict]:
        """Recursively search dictionary for all detection arrays."""
        dets = []
        if isinstance(obj, dict):
            if 'detections' in obj and isinstance(obj['detections'], list):
                dets.extend(obj['detections'])
            for val in obj.values():
                if isinstance(val, dict):
                    dets.extend(self._extract_detections(val))
        return dets

    def compile_evidence(
        self,
        image: np.ndarray,
        tool_results: Dict[str, Any],
        interpretation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compile complete visual evidence inventory (bounding box overlays, GradCAM heatmaps, saliency maps).
        """
        logger.info("Compiling multi-source visual evidence package (USP-2)...")

        visual_outputs = {}
        evidence_records = []

        # 1. Extract all detections recursively across all tool outputs
        all_detections = self._extract_detections(tool_results)
        annotated_image_b64 = None

        logger.info(f"Evidence compilation: Extracted {len(all_detections)} total detections across {len(tool_results)} tools")

        # 2. Extract explicit annotated images or file paths from tool outputs
        def _file_to_b64(path: str) -> Optional[str]:
            if path and os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        return base64.b64encode(f.read()).decode("utf-8")
                except Exception as e:
                    logger.warning(f"Could not read overlay file {path}: {e}")
            return None

        for tool_id, res in tool_results.items():
            if not isinstance(res, dict):
                continue
            out = res.get('output', res)
            if isinstance(out, dict):
                # Extract base64 annotated image if present
                if 'annotated_image' in out and out['annotated_image']:
                    annotated_image_b64 = out['annotated_image']
                    logger.info(f"Found annotated image from '{tool_id}'")

                # Extract map overlay paths from Model 2 / analyzers
                vis_ev = out.get('visual_evidence', {})
                analysis = out.get('analysis', {})

                flood_path = vis_ev.get('flood_overlay_path') or analysis.get('overlay_path')
                if flood_path:
                    b64_str = _file_to_b64(flood_path)
                    if b64_str:
                        visual_outputs['flood_overlay_b64'] = f"data:image/jpeg;base64,{b64_str}"
                        if not annotated_image_b64:
                            annotated_image_b64 = b64_str

                heatmap_path = vis_ev.get('damage_heatmap_path') or analysis.get('heatmap_path')
                if heatmap_path:
                    b64_str = _file_to_b64(heatmap_path)
                    if b64_str:
                        visual_outputs['damage_heatmap_b64'] = f"data:image/jpeg;base64,{b64_str}"

                evac_path = vis_ev.get('evacuation_map_path') or analysis.get('evacuation_map_path')
                if evac_path:
                    b64_str = _file_to_b64(evac_path)
                    if b64_str:
                        visual_outputs['evacuation_map_b64'] = f"data:image/jpeg;base64,{b64_str}"

        # 3. Determine active image to render on
        # If input image is missing or a 0s placeholder, load the latest fetched satellite image tile
        active_image = image
        if active_image is None or (isinstance(active_image, np.ndarray) and np.max(active_image) == 0):
            map_dir = "text_guided_grounding/data/fetched_maps"
            if os.path.exists(map_dir):
                map_files = [os.path.join(map_dir, f) for f in os.listdir(map_dir) if f.startswith("sat_") and f.endswith(".jpg")]
                if map_files:
                    latest_map = max(map_files, key=os.path.getmtime)
                    loaded_img = cv2.imread(latest_map)
                    if loaded_img is not None:
                        active_image = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2RGB)
                        logger.info(f"Loaded satellite map for visual evidence: {latest_map} ({active_image.shape})")

        if active_image is None:
            active_image = np.zeros((512, 512, 3), dtype=np.uint8)

        # 4. Generate Bounding Box Overlay Image
        if annotated_image_b64:
            visual_outputs['roboflow_annotated_image_b64'] = f"data:image/png;base64,{annotated_image_b64}"
            evidence_records.append({
                'evidence_type': 'segmentation_overlay',
                'source': 'model_workflow',
                'b64_key': 'roboflow_annotated_image_b64',
                'description': 'AI-powered segmentation/spectral analysis overlay'
            })

        if all_detections:
            _, bbox_b64 = self.overlay_generator.draw_bounding_boxes(active_image, all_detections)
            visual_outputs['bounding_box_overlay_b64'] = f"data:image/png;base64,{bbox_b64}"
            if 'roboflow_annotated_image_b64' not in visual_outputs:
                visual_outputs['roboflow_annotated_image_b64'] = f"data:image/png;base64,{bbox_b64}"

            evidence_records.append({
                'evidence_type': 'bounding_boxes',
                'source': 'grounding_model',
                'item_count': len(all_detections),
                'b64_key': 'bounding_box_overlay_b64'
            })

        # Extract query for context-aware visualizations
        query = interpretation.get('original_query', '')

        # 5. Generate Query-Aware GradCAM Spatial Attention Heatmap
        _, attention_b64 = self.attention_generator.generate_attention_heatmap(
            active_image, detections=all_detections, query=query
        )
        visual_outputs['spatial_attention_heatmap_b64'] = f"data:image/png;base64,{attention_b64}"
        evidence_records.append({
            'evidence_type': 'gradcam_attention_map',
            'source': 'vision_transformer',
            'b64_key': 'spatial_attention_heatmap_b64'
        })

        # 6. Generate Query-Aware Spatial Saliency Map
        _, saliency_b64 = self.saliency_generator.compute_saliency_map(
            active_image, detections=all_detections, query=query
        )
        visual_outputs['spatial_saliency_map_b64'] = f"data:image/png;base64,{saliency_b64}"
        evidence_records.append({
            'evidence_type': 'saliency_activation_map',
            'source': 'gradient_saliency',
            'b64_key': 'spatial_saliency_map_b64'
        })

        logger.info(f"[OK] Evidence compilation complete: {len(evidence_records)} sources, {len(list(visual_outputs.keys()))} visual outputs")
        logger.info(f"  Visual output keys: {list(visual_outputs.keys())}")

        return {
            'visual_outputs': visual_outputs,
            'evidence_records': evidence_records,
            'total_evidence_sources': len(evidence_records)
        }


