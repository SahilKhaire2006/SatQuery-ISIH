"""
Test building detection with Roboflow integration
"""
import asyncio
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

async def test_building_detector():
    """Test Roboflow building detector"""
    print("=== Testing Roboflow Building Detector ===\n")
    
    # Import detector
    from models.roboflow_building_detector import RoboflowBuildingDetector
    
    # Initialize
    detector = RoboflowBuildingDetector()
    
    if not detector.loaded:
        print("[-] Detector failed to load")
        return
    
    print("[OK] Detector loaded successfully\n")
    
    # Check if test image exists
    test_image_paths = [
        "data/raw/test_satellite_image.png",
        "data/raw/image.png",
        "data/processed/test1.png",
        "data/processed/test2.png",
        "satelite-img.png"
    ]
    
    test_image_path = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image_path = path
            break
    
    if not test_image_path:
        print("[!] No test image found. Please provide a satellite image path.")
        print("Available paths checked:")
        for path in test_image_paths:
            print(f"  - {path}")
        return
    
    print(f"[+] Loading test image: {test_image_path}")
    
    # Load image
    try:
        image = Image.open(test_image_path)
        image_array = np.array(image)
        print(f"   Image size: {image.size}")
        print(f"   Array shape: {image_array.shape}\n")
    except Exception as e:
        print(f"[-] Failed to load image: {e}")
        return
    
    # Run detection
    print("[*] Running building detection...")
    try:
        result = await detector.predict(
            image=image_array,
            query="count buildings in this image",
            parameters={}
        )
        
        print("\n=== Detection Results ===")
        print(f"Status: {result['status']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print(f"\nAnswer: {result['output']['answer']}")
        print(f"Detections: {len(result['output']['detections'])}")
        
        if result['output']['detections']:
            print("\nTop 5 Detections:")
            for i, det in enumerate(result['output']['detections'][:5], 1):
                bbox = det['bbox']
                print(f"  {i}. {det['label']} - confidence: {det['confidence']:.2f} - bbox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        
        if result['status'] in ['ok', 'segmentation_ok']:
            print("\n[OK] Building detection working correctly!")
        else:
            print(f"\n[!] Detection status: {result['status']}")
        
    except Exception as e:
        print(f"[-] Detection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_building_detector())
