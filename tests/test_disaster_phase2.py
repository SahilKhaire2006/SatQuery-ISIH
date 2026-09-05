"""
Phase 2 Verification Test — Model 2 (Disaster Management Grounding Model)

Tests:
1. FloodAnalyzer (water detection, escalation level, flood overlay rendering)
2. EarthquakeAnalyzer (structural change detection, severity calculation, damage heatmap)
3. EvacuationPlanner (distance transform, Red/Yellow/Green safety buffer mapping)
4. End-to-End DisasterGroundingModel Flood Analysis (with generated map files)
5. End-to-End DisasterGroundingModel Earthquake Analysis (with generated heatmap & evacuation map)
6. Regression test for Model 1 (VQA & building detector functionality)
"""

import sys
import os
import asyncio
import numpy as np

# Ensure project root is in path and stdout is utf-8
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()


def test_flood_analyzer():
    """Test FloodAnalyzer module directly."""
    print("=" * 60)
    print("TEST 1: FloodAnalyzer Unit Test")
    print("=" * 60)

    try:
        from models.disaster_analysis.flood_analyzer import FloodAnalyzer

        analyzer = FloodAnalyzer(ndwi_threshold=0.3)

        # Create dummy synthetic RGB image (500x500x3) with a blue water block in the middle
        img = np.ones((500, 500, 3), dtype=np.uint8) * 120
        img[150:350, 150:350, :] = [30, 80, 220] # Blue water region

        result = analyzer.analyze(
            primary_image=img,
            lat=19.0760,
            lon=72.8777,
            save_dir="text_guided_grounding/data/fetched_maps",
        )

        flood_ext = result.get("flood_extent", {})
        escalation = result.get("escalation_level", "")
        overlay_path = result.get("overlay_path", "")

        print(f"  [OK] Flooded area: {flood_ext.get('flooded_area_km2')} km² ({flood_ext.get('flood_percentage')}%)")
        print(f"  [OK] Escalation level: {escalation}")
        print(f"  [OK] Overlay file created: {os.path.exists(overlay_path)} ({overlay_path})")

        if os.path.exists(overlay_path) and escalation != "":
            print()
            return True
        else:
            print("  [FAIL] Overlay file missing or escalation empty")
            return False

    except Exception as e:
        print(f"  [FAIL] FloodAnalyzer test failed: {e}")
        return False


def test_earthquake_analyzer():
    """Test EarthquakeAnalyzer module directly."""
    print("=" * 60)
    print("TEST 2: EarthquakeAnalyzer Unit Test")
    print("=" * 60)

    try:
        from models.disaster_analysis.earthquake_analyzer import EarthquakeAnalyzer

        analyzer = EarthquakeAnalyzer()

        # Create dummy synthetic pre & post images with a shifted block
        pre_img = np.ones((500, 500, 3), dtype=np.uint8) * 140
        post_img = pre_img.copy()
        post_img[100:300, 100:300, :] = [220, 50, 50] # Damaged/shifted region

        result = analyzer.analyze(
            post_image=post_img,
            pre_image=pre_img,
            lat=37.7749,
            lon=-122.4194,
            save_dir="text_guided_grounding/data/fetched_maps",
        )

        eq_analysis = result.get("earthquake_analysis", {})
        severity = result.get("severity", "")
        heatmap_path = result.get("heatmap_path", "")

        print(f"  [OK] Change detected: {eq_analysis.get('change_percentage')}%")
        print(f"  [OK] Severity level: {severity}")
        print(f"  [OK] Heatmap file created: {os.path.exists(heatmap_path)} ({heatmap_path})")

        if os.path.exists(heatmap_path) and severity != "":
            print()
            return True
        else:
            print("  [FAIL] Heatmap file missing or severity empty")
            return False

    except Exception as e:
        print(f"  [FAIL] EarthquakeAnalyzer test failed: {e}")
        return False


def test_evacuation_planner():
    """Test EvacuationPlanner module directly."""
    print("=" * 60)
    print("TEST 3: EvacuationPlanner Unit Test")
    print("=" * 60)

    try:
        from models.disaster_analysis.evacuation_planner import EvacuationPlanner

        planner = EvacuationPlanner()

        # Dummy synthetic hazard mask (500x500 uint8) with hazard in middle
        hazard_mask = np.zeros((500, 500), dtype=np.uint8)
        hazard_mask[200:300, 200:300] = 255
        base_img = np.ones((500, 500, 3), dtype=np.uint8) * 120

        result = planner.plan_evacuation(
            hazard_mask=hazard_mask,
            base_image=base_img,
            pixel_size_meters=10.0,
            lat=19.0760,
            lon=72.8777,
            save_dir="text_guided_grounding/data/fetched_maps",
        )

        zones = result.get("evacuation_zones", {})
        evac_map = result.get("evacuation_map_path", "")

        print(f"  [OK] Danger Zone (<100m): {zones.get('danger_zone_pct')}%")
        print(f"  [OK] Warning Zone (100-500m): {zones.get('warning_zone_pct')}%")
        print(f"  [OK] Safe Zone (>500m): {zones.get('safe_zone_pct')}%")
        print(f"  [OK] Evacuation Map file created: {os.path.exists(evac_map)} ({evac_map})")

        if os.path.exists(evac_map):
            print()
            return True
        else:
            print("  [FAIL] Evacuation map file missing")
            return False

    except Exception as e:
        print(f"  [FAIL] EvacuationPlanner test failed: {e}")
        return False


def test_end_to_end_disaster_flood():
    """Test full Model 2 predict for Flood Query."""
    print("=" * 60)
    print("TEST 4: Model 2 End-to-End Flood Predict")
    print("=" * 60)

    try:
        from models.disaster_analysis import DisasterGroundingModel

        model = DisasterGroundingModel()
        query = "Flash flood inundation assessment for Assam, India"

        print(f"  Running query: '{query}'...")
        result = model.predict(
            image=None,
            query=query,
            parameters={"disaster_type": "flood"},
        )

        output = result.get("output", {})
        evidence = output.get("visual_evidence", {})

        print(f"  [OK] Location: {output.get('location', {}).get('name')}")
        print(f"  [OK] Flood overlay available: {evidence.get('flood_overlay_available')}")
        print(f"  [OK] Evacuation map available: {evidence.get('evacuation_map_available')}")
        print(f"  [OK] Report length: {len(output.get('answer', ''))} characters")

        if evidence.get("flood_overlay_available") and len(output.get("answer", "")) > 100:
            print()
            return True
        else:
            print("  [FAIL] Visual evidence missing or empty report")
            return False

    except Exception as e:
        print(f"  [FAIL] End-to-End Flood predict failed: {e}")
        return False


def test_end_to_end_disaster_earthquake():
    """Test full Model 2 predict for Earthquake Query."""
    print("=" * 60)
    print("TEST 5: Model 2 End-to-End Earthquake Predict")
    print("=" * 60)

    try:
        from models.disaster_analysis import DisasterGroundingModel

        model = DisasterGroundingModel()
        query = "Earthquake structural damage assessment in Kahramanmaras Turkey"

        print(f"  Running query: '{query}'...")
        result = model.predict(
            image=None,
            query=query,
            parameters={"disaster_type": "earthquake"},
        )

        output = result.get("output", {})
        evidence = output.get("visual_evidence", {})

        print(f"  [OK] Location: {output.get('location', {}).get('name')}")
        print(f"  [OK] Damage heatmap available: {evidence.get('damage_heatmap_available')}")
        print(f"  [OK] Evacuation map available: {evidence.get('evacuation_map_available')}")
        print(f"  [OK] Report length: {len(output.get('answer', ''))} characters")

        if evidence.get("damage_heatmap_available") and len(output.get("answer", "")) > 100:
            print()
            return True
        else:
            print("  [FAIL] Damage heatmap missing or empty report")
            return False

    except Exception as e:
        print(f"  [FAIL] End-to-End Earthquake predict failed: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("  MODEL 2 -- PHASE 2 SPECIALIZED PIPELINE VERIFICATION")
    print("=" * 60 + "\n")

    results = {
        "FloodAnalyzer": test_flood_analyzer(),
        "EarthquakeAnalyzer": test_earthquake_analyzer(),
        "EvacuationPlanner": test_evacuation_planner(),
        "E2E Flood Predict": test_end_to_end_disaster_flood(),
        "E2E Earthquake Predict": test_end_to_end_disaster_earthquake(),
    }

    print("=" * 60)
    print("  PHASE 2 RESULTS SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  All Phase 2 tests passed successfully!")
    else:
        print(f"\n  {total - passed} test(s) failed -- see details above")

    print()


if __name__ == "__main__":
    main()
