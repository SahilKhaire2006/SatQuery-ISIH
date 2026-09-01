import sys
import os
import asyncio
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualization.overlay_generator import OverlayGenerator
from visualization.attention_maps import AttentionMapGenerator
from visualization.saliency_maps import SaliencyMapGenerator
from visualization.evidence_compiler import EvidenceCompiler
from agentic_layer.orchestrator import AgenticOrchestrator


def test_overlay_generator():
    gen = OverlayGenerator()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = [{'label': 'building', 'bbox': [10, 10, 50, 50], 'confidence': 0.92}]
    annotated, b64_str = gen.draw_bounding_boxes(img, dets)
    assert annotated.shape == (100, 100, 3)
    assert len(b64_str) > 0


def test_attention_map_generator():
    gen = AttentionMapGenerator()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    heatmap, b64_str = gen.generate_attention_heatmap(img)
    assert heatmap.shape == (100, 100, 3)
    assert len(b64_str) > 0


def test_saliency_map_generator():
    gen = SaliencyMapGenerator()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    saliency, b64_str = gen.compute_saliency_map(img)
    assert saliency.shape == (100, 100, 3)
    assert len(b64_str) > 0


def test_rgba_and_grayscale_image_handling():
    att_gen = AttentionMapGenerator()
    sal_gen = SaliencyMapGenerator()
    over_gen = OverlayGenerator()

    # 4-channel RGBA image test
    rgba_img = np.zeros((100, 100, 4), dtype=np.uint8)
    h_rgba, b64_rgba = att_gen.generate_attention_heatmap(rgba_img)
    assert h_rgba.shape == (100, 100, 3)
    assert len(b64_rgba) > 0

    s_rgba, b64_sal_rgba = sal_gen.compute_saliency_map(rgba_img)
    assert len(b64_sal_rgba) > 0

    o_rgba, b64_over_rgba = over_gen.draw_bounding_boxes(rgba_img, [{'bbox': [0, 0, 50, 50]}])
    assert len(b64_over_rgba) > 0

    # 2D Grayscale image test
    gray_img = np.zeros((100, 100), dtype=np.uint8)
    h_gray, b64_gray = att_gen.generate_attention_heatmap(gray_img)
    assert len(b64_gray) > 0


def test_evidence_compiler():
    compiler = EvidenceCompiler()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    tool_results = {
        'grounding_model': {
            'output': {
                'detections': [{'label': 'structure', 'bbox': [5, 5, 45, 45], 'confidence': 0.88}]
            }
        }
    }
    interpretation = {'task_type': 'grounding'}
    compiled = compiler.compile_evidence(img, tool_results, interpretation)
    assert 'visual_outputs' in compiled
    assert 'bounding_box_overlay_b64' in compiled['visual_outputs']
    assert 'spatial_attention_heatmap_b64' in compiled['visual_outputs']


def test_orchestrator_phase5_integration():
    async def _run():
        orchestrator = AgenticOrchestrator()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', dummy_img)
        image_bytes = buffer.tobytes()

        resp = await orchestrator.process_request(
            session_id="sess_phase5_1",
            query="Locate the building and show attention map",
            image_data=image_bytes,
            image_filename="test.png"
        )
        assert resp['status'] == 'success'
        assert 'visual_evidence' in resp['results']
        assert 'spatial_attention_heatmap_b64' in resp['results']['visual_evidence']

    asyncio.run(_run())


if __name__ == '__main__':
    test_overlay_generator()
    test_attention_map_generator()
    test_saliency_map_generator()
    test_rgba_and_grayscale_image_handling()
    test_evidence_compiler()
    test_orchestrator_phase5_integration()
    print("All Phase 5 Visual Evidence & Audit Trail tests passed successfully!")
