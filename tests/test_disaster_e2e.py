"""
Phase 4 Hardening & End-to-End Pipeline Test — Model 2 Disaster Grounding

Tests:
1. Global Location: Wayanad, Kerala (Flood Grounding)
2. Global Location: Turkey (Earthquake Structural Damage)
3. Global Location: Pakistan (Flood Assessment)
4. Global Location: Assam, India (Monsoon Inundation)
5. Edge Case: Invalid/Ambiguous Location Name ("xyz999nonexistentlocation")
6. Edge Case: Sentinel Hub Unconfigured Fallback (Esri World Imagery)
"""

import sys
import os
import asyncio

# Ensure project root is in path and stdout is utf-8
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()


def test_global_locations():
    """Test Model 2 predict across known historical disaster locations."""
    print("=" * 60)
    print("TEST 1: Global Location Predict Suite")
    print("=" * 60)

    from models.disaster_analysis import DisasterGroundingModel

    model = DisasterGroundingModel()

    locations = [
        ("Wayanad, Kerala", "flood", "Show flood extent in Wayanad, Kerala"),
        ("Kahramanmaras, Turkey", "earthquake", "Earthquake damage assessment in Turkey"),
        ("Sindh, Pakistan", "flood", "Flash flood inundation in Sindh, Pakistan"),
        ("Assam, India", "flood", "Monsoon flood extent assessment in Assam"),
    ]

    all_passed = True

    for loc_name, d_type, query in locations:
        print(f"\n  Running: '{query}'...")
        try:
            result = model.predict(
                image=None,
                query=query,
                parameters={"disaster_type": d_type},
            )

            output = result.get("output", {})
            conf = result.get("confidence", 0.0)
            answer = output.get("answer", "")
            loc = output.get("location", {}).get("name", "Unknown")

            print(f"    [OK] Resolved: {loc[:45]}")
            print(f"    [OK] Confidence: {conf}")
            print(f"    [OK] Report generated: {len(answer)} chars")

            if conf > 0 and len(answer) > 100 and "could not be completed" not in answer.lower():
                print(f"    [PASS] Predict for {loc_name} succeeded!")
            else:
                print(f"    [FAIL] Predict for {loc_name} failed!")
                all_passed = False

        except Exception as e:
            print(f"    [FAIL] Exception for {loc_name}: {e}")
            all_passed = False

    print()
    return all_passed


def test_invalid_location_edge_case():
    """Test edge case handling when location cannot be geocoded."""
    print("=" * 60)
    print("TEST 2: Edge Case — Invalid Location Resolution")
    print("=" * 60)

    from models.disaster_analysis import DisasterGroundingModel

    model = DisasterGroundingModel()
    query = "Show flood extent in xyz999nonexistentlocation"

    print(f"  Running query with invalid location: '{query}'...")
    result = model.predict(
        image=None,
        query=query,
        parameters={"disaster_type": "flood"},
    )

    output = result.get("output", {})
    answer = output.get("answer", "")
    error = output.get("error", "")

    print(f"  Output error message: {error or answer[:120]}")

    if "location resolution failed" in answer.lower() or "location resolution failed" in error.lower() or "could not resolve" in answer.lower() or "invalid" in answer.lower():
        print("  [PASS] Invalid location handled gracefully with user guidance!")
        print()
        return True
    else:
        print("  [FAIL] Did not return expected location error response")
        print()
        return False


def test_sentinel_fallback_edge_case():
    """Test edge case handling when Sentinel Hub is unconfigured (Esri fallback)."""
    print("=" * 60)
    print("TEST 3: Edge Case — Sentinel Hub Fallback to Esri")
    print("=" * 60)

    from geospatial.imagery_router import ImageryRouter

    router = ImageryRouter()
    # Explicitly test Esri fallback
    result = router._fetch_esri_fallback(
        lat=28.6139,
        lon=77.2090,
        area_meters=1000.0,
        result={"primary_image": None, "metadata": {"sources_used": []}},
    )

    img = result.get("primary_image")
    sources = result["metadata"]["sources_used"]

    print(f"  [OK] Image shape: {img.shape if img is not None else None}")
    print(f"  [OK] Sources used: {sources}")

    if img is not None and "Esri World Imagery (fallback)" in sources:
        print("  [PASS] Esri fallback router functional!")
        print()
        return True
    else:
        print("  [FAIL] Esri fallback failed")
        print()
        return False


def main():
    print("\n" + "=" * 60)
    print("  MODEL 2 -- PHASE 4 HARDENING & END-TO-END VERIFICATION")
    print("=" * 60 + "\n")

    results = {
        "Global Locations Suite": test_global_locations(),
        "Invalid Location Edge Case": test_invalid_location_edge_case(),
        "Sentinel/Esri Fallback Edge Case": test_sentinel_fallback_edge_case(),
    }

    print("=" * 60)
    print("  PHASE 4 HARDENING SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 Phase 4 Hardening & E2E Verification Complete — Model 2 Production Ready!")
    else:
        print(f"\n  ⚠ {total - passed} test(s) failed -- see details above")

    print()


if __name__ == "__main__":
    main()
