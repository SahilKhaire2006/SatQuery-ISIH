"""
NASA Earthdata & FIRMS Unit & Integration Test

Tests:
1. NASA FIRMS Active Thermal Hotspot REST API query
2. NASA Earthdata CMR Granule Search REST API query
3. ImageryRouter integration with NASA Earthdata & FIRMS
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


def test_nasa_firms():
    """Test NASA FIRMS active thermal hotspot API query."""
    print("=" * 60)
    print("TEST 1: NASA FIRMS Active Fire API Query")
    print("=" * 60)

    try:
        from geospatial.nasa_earthdata_fetcher import fetch_nasa_firms_hotspots

        # Query known active fire region or California coordinates (37.7749, -122.4194)
        result = fetch_nasa_firms_hotspots(lat=37.7749, lon=-122.4194, day_range=3)

        source = result.get("source", "")
        count = result.get("hotspot_count", 0)

        print(f"  [OK] Source: {source}")
        print(f"  [OK] Active thermal hotspots detected: {count}")

        if source and isinstance(count, int):
            print("  [PASS] NASA FIRMS query successful!")
            print()
            return True
        else:
            print("  [FAIL] NASA FIRMS query failed")
            print()
            return False

    except Exception as e:
        print(f"  [FAIL] NASA FIRMS test failed: {e}")
        print()
        return False


def test_nasa_cmr():
    """Test NASA Earthdata CMR granule search API query."""
    print("=" * 60)
    print("TEST 2: NASA Earthdata CMR Granule Search")
    print("=" * 60)

    try:
        from geospatial.nasa_earthdata_fetcher import fetch_nasa_cmr_granules

        result = fetch_nasa_cmr_granules(lat=28.6139, lon=77.2090, short_name="MOD09GA")

        source = result.get("source", "")
        count = result.get("granule_count", 0)

        print(f"  [OK] Source: {source}")
        print(f"  [OK] Granules found: {count}")

        if source and isinstance(count, int):
            print("  [PASS] NASA CMR query successful!")
            print()
            return True
        else:
            print("  [FAIL] NASA CMR query failed")
            print()
            return False

    except Exception as e:
        print(f"  [FAIL] NASA CMR test failed: {e}")
        print()
        return False


def test_imagery_router_nasa():
    """Test ImageryRouter includes NASA Earthdata & FIRMS."""
    print("=" * 60)
    print("TEST 3: ImageryRouter NASA Integration")
    print("=" * 60)

    try:
        from geospatial.imagery_router import ImageryRouter

        router = ImageryRouter()
        res = router.fetch_disaster_imagery(
            lat=37.7749, lon=-122.4194, disaster_type="wildfire"
        )

        sources = res.get("metadata", {}).get("sources_used", [])
        print(f"  [OK] Sources used for wildfire: {sources}")

        if any("NASA" in s for s in sources):
            print("  [PASS] NASA Earthdata & FIRMS active in ImageryRouter!")
            print()
            return True
        else:
            print("  [FAIL] NASA source not recorded in ImageryRouter output")
            print()
            return False

    except Exception as e:
        print(f"  [FAIL] ImageryRouter NASA test failed: {e}")
        print()
        return False


def main():
    print("\n" + "=" * 60)
    print("  NASA EARTHDATA & FIRMS INTEGRATION VERIFICATION")
    print("=" * 60 + "\n")

    results = {
        "NASA FIRMS Query": test_nasa_firms(),
        "NASA CMR Search": test_nasa_cmr(),
        "ImageryRouter Integration": test_imagery_router_nasa(),
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
        print("\n  🎉 NASA Earthdata & FIRMS Integration Fully Complete!")
    else:
        print(f"\n  ⚠ {total - passed} test(s) failed")

    print()


if __name__ == "__main__":
    main()
