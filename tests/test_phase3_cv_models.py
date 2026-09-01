import sys
import os
import asyncio
import numpy as np
import cv2
import pytest

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_manager import ModelManager
from models.vqa_model import VQAModel
from models.grounding_model import GroundingModel
from models.change_detection_model import ChangeDetectionModel
from models.sar_fusion_model import SARFusionModel
from agentic_layer.orchestrator import AgenticOrchestrator


def test_model_manager():
    manager = ModelManager()
    assert manager is not None
    dummy_np = np.zeros((50, 50, 3), dtype=np.uint8)
    pil_img = manager.numpy_to_pil(dummy_np)
    assert pil_img.size == (50, 50)


def test_vqa_model():
    vqa = VQAModel()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = vqa.predict(dummy_img, "Is there a building?")
    assert 'output' in res
    assert 'answer' in res['output']
    assert res['confidence'] > 0.0


def test_grounding_model():
    grounding = GroundingModel()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = grounding.predict(dummy_img, "Locate roads")
    assert 'output' in res
    assert 'detections' in res['output']
    assert len(res['output']['detections']) > 0
    assert 'bbox' in res['output']['detections'][0]


def test_change_detection_model():
    cd = ChangeDetectionModel()
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 100
    res = cd.predict(img1, "Detect urban changes", parameters={'second_image': img2})
    assert 'output' in res
    assert 'change_percentage' in res['output']
    assert res['output']['change_percentage'] > 0.0


def test_sar_fusion_model():
    sar = SARFusionModel()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = sar.predict(dummy_img, "Fuse Sentinel-1 radar features")
    assert 'output' in res
    assert 'fusion_info' in res['output']
    assert 'sar_features' in res['output']['fusion_info']


def test_orchestrator_phase3_integration():
    async def _run():
        orchestrator = AgenticOrchestrator()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', dummy_img)
        image_bytes = buffer.tobytes()

        # Test VQA query
        vqa_resp = await orchestrator.process_request(
            session_id="sess_phase3_1",
            query="What type of terrain is shown?",
            image_data=image_bytes,
            image_filename="test.png"
        )
        assert vqa_resp['status'] == 'success'

        # Test SAR query (USP-1)
        sar_resp = await orchestrator.process_request(
            session_id="sess_phase3_2",
            query="Analyze Sentinel-1 SAR radar imagery",
            image_data=image_bytes,
            image_filename="test.png"
        )
        assert sar_resp['status'] == 'success'
        assert any(t['tool_id'] == 'sar_fusion_model' for t in sar_resp['audit_log']['selected_tools'])

    asyncio.run(_run())


if __name__ == "__main__":
    test_model_manager()
    test_vqa_model()
    test_grounding_model()
    test_change_detection_model()
    test_sar_fusion_model()
    test_orchestrator_phase3_integration()
    print("All Phase 3 Computer Vision Model tests passed successfully!")
