"""
Test script to verify Roboflow integration and annotated image flow
"""
import asyncio
import sys
import os
import numpy as np
from PIL import Image
import json
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

sys.path.insert(0, '.')

from models.roboflow_building_detector import RoboflowBuildingDetector
from visualization.evidence_compiler import EvidenceCompiler
from utils.logger import setup_logger

logger = setup_logger(__name__)


async def test_roboflow_detection():
    """Test Roboflow building detection"""
    logger.info("=" * 60)
    logger.info("TESTING ROBOFLOW INTEGRATION")
    logger.info("=" * 60)
    
    # Load test image
    try:
        img_path = 'data/raw/test_satellite_image.png'
        pil_img = Image.open(img_path)
        img_array = np.array(pil_img)
        logger.info(f"✓ Loaded test image: {img_path} (shape: {img_array.shape})")
    except Exception as e:
        logger.error(f"✗ Failed to load test image: {e}")
        return
    
    # Initialize detector
    detector = RoboflowBuildingDetector()
    if not detector.loaded:
        logger.error("✗ Roboflow detector failed to initialize")
        return
    
    logger.info("✓ Roboflow detector initialized")
    
    # Run detection
    logger.info("\n--- Running detection ---")
    result = await detector.predict(img_array, 'count buildings', {})
    
    # Analyze results
    logger.info("\n--- Detection Results ---")
    logger.info(f"Status: {result.get('status')}")
    logger.info(f"Confidence: {result.get('confidence'):.3f}")
    logger.info(f"Execution time: {result.get('execution_time'):.2f}s")
    
    output = result.get('output', {})
    logger.info(f"\nOutput keys: {list(output.keys())}")
    logger.info(f"Answer: {output.get('answer')}")
    logger.info(f"Detections count: {len(output.get('detections', []))}")
    
    # Print first few detections if available
    if output.get('detections'):
        logger.info("\nFirst 3 detections:")
        for idx, det in enumerate(output['detections'][:3]):
            logger.info(f"  {idx+1}. bbox={det.get('bbox')}, conf={det.get('confidence'):.3f}, label={det.get('label')}")
    
    logger.info(f"Has annotated_image: {output.get('annotated_image') is not None}")
    logger.info(f"Has segmentation: {output.get('has_segmentation', False)}")
    logger.info(f"Has bounding boxes: {output.get('has_bounding_boxes', False)}")
    
    if output.get('annotated_image'):
        img_len = len(output['annotated_image'])
        logger.info(f"Annotated image length: {img_len} chars")
        logger.info(f"Annotated image preview: {output['annotated_image'][:100]}...")
    
    # Test evidence compiler
    logger.info("\n--- Testing Evidence Compiler ---")
    evidence_compiler = EvidenceCompiler()
    
    tool_results = {
        'building_detector': result
    }
    
    evidence = evidence_compiler.compile_evidence(
        image=img_array,
        tool_results=tool_results,
        interpretation={'task_type': 'building_detection'}
    )
    
    logger.info(f"\nEvidence compilation results:")
    logger.info(f"Visual output keys: {list(evidence['visual_outputs'].keys())}")
    logger.info(f"Evidence records: {len(evidence['evidence_records'])}")
    
    for record in evidence['evidence_records']:
        logger.info(f"  - {record['evidence_type']} from {record['source']}")
    
    # Check for Roboflow annotated image
    if 'roboflow_annotated_image_b64' in evidence['visual_outputs']:
        roboflow_img = evidence['visual_outputs']['roboflow_annotated_image_b64']
        logger.info(f"\n✓ SUCCESS: Roboflow annotated image found in visual outputs!")
        logger.info(f"  Key: roboflow_annotated_image_b64")
        logger.info(f"  Length: {len(roboflow_img)} chars")
        logger.info(f"  Has data URI prefix: {roboflow_img.startswith('data:image/png;base64,')}")
    else:
        logger.warning(f"\n✗ WARNING: Roboflow annotated image NOT found in visual outputs")
        logger.warning(f"  Available keys: {list(evidence['visual_outputs'].keys())}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Detector loaded: {'✓' if detector.loaded else '✗'}")
    logger.info(f"Detection executed: {'✓' if result.get('status') != 'failed' else '✗'}")
    logger.info(f"Annotated image returned: {'✓' if output.get('annotated_image') else '✗'}")
    logger.info(f"Evidence compiled: {'✓' if evidence else '✗'}")
    logger.info(f"Roboflow image in visual_outputs: {'✓' if 'roboflow_annotated_image_b64' in evidence.get('visual_outputs', {}) else '✗'}")
    logger.info("=" * 60)


if __name__ == '__main__':
    asyncio.run(test_roboflow_detection())
