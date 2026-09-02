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

        # 1. Check for Roboflow or Grounding Bounding Box Detections + Annotated Images
        all_detections = []
        annotated_image_b64 = None
        annotated_image_type = None  # Track type: 'building' or 'water'
        
        logger.info(f"Evidence compilation: Processing {len(tool_results)} tool results")
        
        for tool_id, res in tool_results.items():
            logger.debug(f"Processing tool: {tool_id}, result type: {type(res)}")
            if isinstance(res, dict):
                out = res.get('output', res)
                if isinstance(out, dict):
                    # Extract detections
                    if 'detections' in out:
                        det_count = len(out['detections'])
                        all_detections.extend(out['detections'])
                        logger.info(f"Tool '{tool_id}' contributed {det_count} detections (total: {len(all_detections)})")
                    
                    # Extract Roboflow annotated image (building or water body)
                    if 'annotated_image' in out and out['annotated_image']:
                        annotated_image_b64 = out['annotated_image']
                        # Determine type based on tool
                        if 'waterbody' in tool_id.lower() or 'water' in tool_id.lower():
                            annotated_image_type = 'water'
                        else:
                            annotated_image_type = 'building'
                        logger.info(f"Found Roboflow {annotated_image_type} annotated image from '{tool_id}' (length: {len(annotated_image_b64)} chars)")
                        logger.info(f"Annotated image preview: {annotated_image_b64[:100]}...")

        # If we have Roboflow annotated image, use it directly
        if annotated_image_b64:
            visual_outputs['roboflow_annotated_image_b64'] = f"data:image/png;base64,{annotated_image_b64}"
            visual_outputs['roboflow_image_type'] = annotated_image_type  # 'building' or 'water'
            logger.info(f"Added Roboflow {annotated_image_type} annotated image to visual outputs (key: roboflow_annotated_image_b64)")
            evidence_records.append({
                'evidence_type': f'roboflow_{annotated_image_type}_segmentation',
                'source': 'roboflow_workflow',
                'b64_key': 'roboflow_annotated_image_b64',
                'description': f'AI-powered {annotated_image_type} segmentation from Roboflow workflow'
            })
        
        # If we have detection boxes, draw them
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

        logger.info(f"✓ Evidence compilation complete: {len(evidence_records)} sources, {len(list(visual_outputs.keys()))} visual outputs")
        logger.info(f"  Visual output keys: {list(visual_outputs.keys())}")
        
        return {
            'visual_outputs': visual_outputs,
            'evidence_records': evidence_records,
            'total_evidence_sources': len(evidence_records)
        }
