"""
Phase 1 Verification Test — Model 2 (Disaster Grounding)

Tests:
1. Sentinel fetcher module imports correctly
2. Imagery router initializes and routes correctly
3. Disaster grounding model instantiates
4. Query interpreter classifies disaster intents
5. Tool selector routes disaster queries
6. Esri fallback works when Sentinel Hub is unconfigured
7. Full pipeline end-to-end (text-only disaster query)
"""

import sys
import os
import asyncio

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def test_imports():
    """Test all new modules import without errors."""
    print("=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)

    try:
        from geospatial.sentinel_fetcher import (
            is_sentinel_hub_available,
            fetch_sentinel2_rgb,
            fetch_sentinel1_sar,
            fetch_ndwi_layer,
            fetch_multi_temporal,
            SENTINELHUB_AVAILABLE,
        )
        print(f"  [OK] sentinel_fetcher imported (sentinelhub package: {'available' if SENTINELHUB_AVAILABLE else 'NOT installed'})")
    except Exception as e:
        print(f"  [FAIL] sentinel_fetcher import FAILED: {e}")
        return False

    try:
        from geospatial.imagery_router import ImageryRouter, fetch_disaster_imagery
        print("  [OK] imagery_router imported")
    except Exception as e:
        print(f"  [FAIL] imagery_router import FAILED: {e}")
        return False

    try:
        from models.disaster_analysis import DisasterGroundingModel
        print("  [OK] disaster_grounding_model imported")
    except Exception as e:
        print(f"  [FAIL] disaster_grounding_model import FAILED: {e}")
        return False

    print()
    return True


def test_sentinel_hub_config():
    """Test Sentinel Hub configuration."""
    print("=" * 60)
    print("TEST 2: Sentinel Hub Configuration")
    print("=" * 60)

    from geospatial.sentinel_fetcher import is_sentinel_hub_available, SENTINELHUB_AVAILABLE

    print(f"  sentinelhub package installed: {SENTINELHUB_AVAILABLE}")

    client_id = os.getenv("COPERNICUS_CLIENT_ID", "")
    client_secret = os.getenv("COPERNICUS_CLIENT_SECRET", "")
    print(f"  COPERNICUS_CLIENT_ID set: {bool(client_id)}")
    print(f"  COPERNICUS_CLIENT_SECRET set: {bool(client_secret)}")

    available = is_sentinel_hub_available()
    print(f"  Sentinel Hub fully configured: {available}")

    if not available:
        print("  [WARN] Sentinel Hub not configured -- Esri fallback will be used")
        print("    To enable: Register at https://dataspace.copernicus.eu/")
        print("    Then add COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET to .env")

    print()
    return True


def test_imagery_router():
    """Test imagery router initialization and Esri fallback."""
    print("=" * 60)
    print("TEST 3: Imagery Router (Esri Fallback)")
    print("=" * 60)

    from geospatial.imagery_router import ImageryRouter

    router = ImageryRouter()
    print(f"  [OK] ImageryRouter initialized")
    print(f"    Sentinel Hub available: {router._sentinel_available}")
    print(f"    Esri available: {router._esri_available}")

    # Test Esri fallback with a known location (Mumbai, India)
    print("\n  Fetching Esri fallback imagery for Mumbai (19.0760, 72.8777)...")
    try:
        result = router._fetch_esri_fallback(
            lat=19.0760,
            lon=72.8777,
            area_meters=1000.0,
            result={
                "primary_image": None,
                "metadata": {"sources_used": []},
            },
        )
        img = result.get("primary_image")
        if img is not None:
            print(f"  [OK] Esri image fetched: shape={img.shape}, dtype={img.dtype}")
            print(f"    Sources: {result['metadata']['sources_used']}")
        else:
            print("  [FAIL] Esri fallback returned None")
            return False
    except Exception as e:
        print(f"  [FAIL] Esri fallback FAILED: {e}")
        return False

    print()
    return True


def test_disaster_model_init():
    """Test DisasterGroundingModel instantiation."""
    print("=" * 60)
    print("TEST 4: Disaster Grounding Model Initialization")
    print("=" * 60)

    from models.disaster_analysis import DisasterGroundingModel

    model = DisasterGroundingModel()
    print(f"  [OK] Model instantiated: {model.model_name}")
    print(f"    Loaded: {model.loaded}")

    print()
    return True


def test_disaster_intent_classification():
    """Test query interpreter disaster intent classification."""
    print("=" * 60)
    print("TEST 5: Disaster Intent Classification")
    print("=" * 60)

    from agentic_layer.query_interpreter import QueryInterpreter

    interp = QueryInterpreter.__new__(QueryInterpreter)

    test_cases = [
        ("Show flood extent in Wayanad Kerala", "disaster_flood"),
        ("Earthquake damage assessment in Turkey", "disaster_earthquake"),
        ("Tsunami evacuation plan for Chennai coast", "disaster_general"),
        ("How many buildings in this image", "building_detection"),
        ("Detect water bodies near Mumbai", "water_detection"),
        ("Flash flood inundation in Assam", "disaster_flood"),
        ("Cyclone damage relief Odisha", "disaster_general"),
        ("Show vegetation in Delhi", "vegetation_detection"),
        ("Landslide risk assessment Uttarakhand", "disaster_general"),
    ]

    all_passed = True
    for query, expected_intent in test_cases:
        result = interp._classify_intent(query, {"task_type": "vqa"})
        status = "[OK]" if result == expected_intent else "[FAIL]"
        if result != expected_intent:
            all_passed = False
        print(f"  {status} \"{query[:50]}...\" -> {result} (expected: {expected_intent})")

    print()
    return all_passed


def test_tool_selector_routing():
    """Test tool selector routes disaster intents correctly."""
    print("=" * 60)
    print("TEST 6: Tool Selector Disaster Routing")
    print("=" * 60)

    async def run_test():
        from agentic_layer.tool_selector import ToolSelector
        selector = ToolSelector()

        # Test flood routing
        interpretation = {
            "intent": "disaster_flood",
            "task_type": "disaster_flood",
            "original_query": "Show flood extent in Kerala",
            "parameters": {},
            "spatial_metadata": {},
            "temporal_aspects": {},
        }

        tools = await selector.select_tools(interpretation, "disaster_flood")
        print(f"  Flood query -> tools: {[t['tool_id'] for t in tools]}")

        if any(t["tool_id"] == "disaster_grounding_model" for t in tools):
            print("  [OK] Correctly routed to disaster_grounding_model")
        else:
            print("  [FAIL] Did NOT route to disaster_grounding_model")
            return False

        # Test earthquake routing
        interpretation["intent"] = "disaster_earthquake"
        interpretation["original_query"] = "Earthquake damage Turkey"
        tools = await selector.select_tools(interpretation, "disaster_earthquake")
        print(f"  Earthquake query -> tools: {[t['tool_id'] for t in tools]}")

        if any(t["tool_id"] == "disaster_grounding_model" for t in tools):
            print("  [OK] Correctly routed to disaster_grounding_model")
        else:
            print("  [FAIL] Did NOT route to disaster_grounding_model")
            return False

        return True

    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run_test())
        loop.close()
    except Exception as e:
        print(f"  [FAIL] Tool selector test FAILED: {e}")
        result = False

    print()
    return result


def test_disaster_model_esri_predict():
    """Test end-to-end predict with Esri fallback (no Sentinel Hub needed)."""
    print("=" * 60)
    print("TEST 7: End-to-End Predict (Esri Fallback)")
    print("=" * 60)

    from models.disaster_analysis import DisasterGroundingModel
    import numpy as np

    model = DisasterGroundingModel()

    # Simple text-only query -- should geocode + fetch from Esri + analyze
    print("  Running predict: 'Show flood extent in Mumbai, India'...")
    result = model.predict(
        image=None,
        query="Show flood extent in Mumbai, India",
        parameters={"disaster_type": "flood"},
    )

    output = result.get("output", {})
    confidence = result.get("confidence", 0)

    print(f"  Confidence: {confidence}")
    print(f"  Model: {output.get('model', 'N/A')}")
    print(f"  Location: {output.get('location', {}).get('name', 'N/A')}")
    print(f"  Imagery sources: {output.get('imagery_source', {}).get('providers', [])}")
    
    # Check analysis
    analysis = output.get("analysis", {})
    flood_ext = analysis.get("flood_extent", {})
    if flood_ext:
        print(f"  Flood analysis: {flood_ext}")

    answer = output.get("answer", "")
    if answer and "could not" not in answer.lower():
        print(f"  [OK] Disaster report generated ({len(answer)} chars)")
        # Print first 200 chars of the report
        print(f"  Report preview: {answer[:200]}...")
    else:
        print(f"  [WARN] Report: {answer[:200]}")

    print()
    return confidence > 0


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("\n" + "=" * 60)
    print("  MODEL 2 -- PHASE 1 VERIFICATION TESTS")
    print("=" * 60 + "\n")

    results = {
        "Imports": test_imports(),
        "Sentinel Hub Config": test_sentinel_hub_config(),
        "Imagery Router": test_imagery_router(),
        "Model Init": test_disaster_model_init(),
        "Intent Classification": test_disaster_intent_classification(),
        "Tool Routing": test_tool_selector_routing(),
        "E2E Predict (Esri)": test_disaster_model_esri_predict(),
    }

    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  All Phase 1 tests passed!")
    else:
        print(f"\n  {total - passed} test(s) failed -- see details above")

    print()


if __name__ == "__main__":
    main()
