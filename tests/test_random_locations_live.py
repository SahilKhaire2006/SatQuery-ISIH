"""
Random Location & Coordinate Live System Verification

Tests Model 2 via AgenticOrchestrator across diverse random coordinates & region names:
1. Coordinates: (13.0827, 80.2707) — Chennai Coast, India
2. Region: "Valencia, Spain" — Flood inundation query
3. Region: "Maui, Hawaii" — Wildfire & NASA FIRMS thermal query
4. Coordinates: (-33.8688, 151.2093) — Sydney, Australia
5. Region: "Kathmandu, Nepal" — Seismic hazard query
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


async def run_live_tests():
    print("\n" + "=" * 65)
    print("  SATQUERY MODEL 2 -- RANDOM LOCATION & COORDINATES LIVE TEST")
    print("=" * 65 + "\n")

    from agentic_layer.orchestrator import AgenticOrchestrator
    orchestrator = AgenticOrchestrator()

    test_cases = [
        {
            "name": "Coordinates: (13.0827, 80.2707) — Chennai Coast, India",
            "query": "Show flood extent and evacuation zones at 13.0827, 80.2707",
        },
        {
            "name": "Region: 'Valencia, Spain'",
            "query": "Flash flood inundation assessment in Valencia, Spain",
        },
        {
            "name": "Region: 'Maui, Hawaii' (Wildfire + NASA FIRMS)",
            "query": "Wildfire thermal hotspot and damage analysis in Maui, Hawaii",
        },
        {
            "name": "Coordinates: (-33.8688, 151.2093) — Sydney, Australia",
            "query": "Coastal inundation risk assessment at -33.8688, 151.2093",
        },
        {
            "name": "Region: 'Kathmandu, Nepal'",
            "query": "Earthquake structural damage assessment in Kathmandu, Nepal",
        },
    ]

    all_passed = True

    for i, test in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Testing {test['name']}...")
        print(f"    Query: \"{test['query']}\"")
        try:
            response = await orchestrator.process_request(
                session_id=f"live_test_{i}",
                query=test["query"],
                image_data=b"",
                image_filename="",
            )

            status = response.get("status", "failed")
            explanation = response.get("explanation", "")
            results = response.get("results", {})

            # Extract model 2 output details
            d_out = results.get("disaster_grounding_model", {}).get("output", {})
            loc_resolved = d_out.get("location", {}).get("name", "N/A")
            sources = d_out.get("imagery_source", {}).get("providers", [])
            evidence = d_out.get("visual_evidence", {})

            print(f"    [OK] Status: {status}")
            print(f"    [OK] Resolved Location: {loc_resolved}")
            print(f"    [OK] Imagery Providers: {sources}")
            print(f"    [OK] Report Length: {len(explanation)} characters")
            print(f"    [OK] Evidence Overlays: {list(evidence.keys())}")

            if status == "success" and len(explanation) > 150:
                print(f"    [PASS] Test {i} Succeeded!\n")
            else:
                print(f"    [FAIL] Test {i} Failed!\n")
                all_passed = False

        except Exception as e:
            print(f"    [FAIL] Test {i} raised exception: {e}\n")
            all_passed = False

    print("=" * 65)
    if all_passed:
        print("  🎉 ALL RANDOM LOCATIONS & COORDINATES PROCESSED PERFECTLY!")
    else:
        print("  ⚠ Some location queries failed.")
    print("=" * 65 + "\n")

    return all_passed


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    success = loop.run_until_complete(run_live_tests())
    loop.close()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
