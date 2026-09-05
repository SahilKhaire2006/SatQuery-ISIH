"""
End-to-End Orchestrator Pipeline Test — Model 2 Disaster Grounding

Tests full orchestrator execution for text-only disaster queries:
1. "Show flood extent and evacuation plan in Wayanad, Kerala"
2. "Earthquake damage assessment in Turkey"
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


async def run_orchestrator_test():
    print("\n" + "=" * 60)
    print("  MODEL 2 END-TO-END ORCHESTRATOR PIPELINE VERIFICATION")
    print("=" * 60 + "\n")

    from agentic_layer.orchestrator import AgenticOrchestrator

    orchestrator = AgenticOrchestrator()

    queries = [
        "Show flood extent and evacuation plan in Wayanad, Kerala",
        "Earthquake damage assessment in Turkey",
    ]

    all_passed = True

    for i, query in enumerate(queries, 1):
        print(f"\n--- Orchestrator Test {i}: '{query}' ---")
        try:
            # Process query with AgenticOrchestrator
            response = await orchestrator.process_request(
                session_id="test_session_phase3",
                query=query,
                image_data=b"",
                image_filename="",
            )

            status = response.get("status", "failed")
            answer = response.get("explanation", response.get("answer", ""))
            results = response.get("results", {})
            
            # Check visual evidence in results
            model_out = results.get("disaster_grounding_model", {}).get("output", {})
            evidence = model_out.get("visual_evidence", results.get("visual_evidence", {}))

            print(f"  [OK] Status: {status}")
            print(f"  [OK] Answer length: {len(answer)} characters")
            print(f"  [OK] Visual evidence overlays: {list(evidence.keys())}")

            if status in ["success", "completed"] and len(answer) > 100:
                print(f"  [PASS] Test {i} succeeded!")
            else:
                print(f"  [FAIL] Test {i} failed -- status={status}, answer_len={len(answer)}")
                all_passed = False

        except Exception as e:
            print(f"  [FAIL] Test {i} raised exception: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("  🎉 All Orchestrator End-to-End tests passed successfully!")
    else:
        print("  ⚠ Some Orchestrator tests failed.")
    print("=" * 60 + "\n")

    return all_passed


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    success = loop.run_until_complete(run_orchestrator_test())
    loop.close()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
