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

        # 1. Check for Grounding Bounding Box Detections
        all_detections = []
        for tool_id, res in tool_results.items():
            if isinstance(res, dict):
                out = res.get('output', res)
                if isinstance(out, dict) and 'detections' in out:
                    all_detections.extend(out['detections'])

        if all_detections:
            _, bbox_b64 = self.overlay_generator.draw_bounding_boxes(image, all_detections)
            visual_outputs['bounding_box_overlay_b64'] = f"data:image/png;base64,{bbox_b64}"
            evidence_records.append({
                'evidence_type': 'bounding_boxes',
                'source': 'grounding_model',
                'item_count': len(all_detections),
                'b64_key': 'bounding_box_overlay_b64'
            })

        # 2. Generate GradCAM Spatial Attention Heatmap
        _, attention_b64 = self.attention_generator.generate_attention_heatmap(image)
        visual_outputs['spatial_attention_heatmap_b64'] = f"data:image/png;base64,{attention_b64}"
        evidence_records.append({
            'evidence_type': 'gradcam_attention_map',
            'source': 'vision_transformer',
            'b64_key': 'spatial_attention_heatmap_b64'
        })

        # 3. Generate Spatial Activation Saliency Map
        _, saliency_b64 = self.saliency_generator.compute_saliency_map(image)
        visual_outputs['spatial_saliency_map_b64'] = f"data:image/png;base64,{saliency_b64}"
        evidence_records.append({
            'evidence_type': 'saliency_activation_map',
            'source': 'gradient_saliency',
            'b64_key': 'spatial_saliency_map_b64'
        })

        return {
            'visual_outputs': visual_outputs,
            'evidence_records': evidence_records,
            'total_evidence_sources': len(evidence_records)
        }
