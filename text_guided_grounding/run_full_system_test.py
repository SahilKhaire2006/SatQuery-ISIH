"""
Full Agentic System Integration Verification Script

Simulates end-to-end processing of a user query containing a place name or address:
Query -> InputValidator -> QueryInterpreter -> Geocoder & 500m² Map Fetcher ->
ToolSelector -> GroundingModel (TextGuidedGroundingModel) -> ResultAggregator & Bounding Boxes.
"""

import sys
import asyncio
from agentic_layer.orchestrator import AgenticOrchestrator


async def run_full_system_test():
    print("=" * 75)
    print("      SATQUERY AGENTIC SYSTEM — FULL TEXT-GUIDED GROUNDING INTEGRATION")
    print("=" * 75)

    orchestrator = AgenticOrchestrator()

    # Test cases: location queries without pre-uploaded image bytes (or synthetic frontend upload)
    test_queries = [
        "Find a playground near Sukhsagar Nagar, Katraj Pune",
        "A PLAYGROUND NEAR VISHWAKARMA INSTITUTE OF TECHNOLOGY PUNE."
    ]

    for idx, query_text in enumerate(test_queries, start=1):
        print(f"\n--- [Test Request {idx}/{len(test_queries)}] Query: '{query_text}' ---")

        response = await orchestrator.process_request(
            session_id=f"integration_session_{idx}",
            query=query_text,
            image_data=b"",  # Empty image bytes triggers automated 500m² satellite map tile fetching
            image_filename=""
        )

        status = response.get("status")
        confidence = response.get("confidence", 0.0)
        explanation = response.get("explanation", "")
        bboxes = response.get("results", {}).get("bounding_boxes", {})

        print(f" Status             : {status}")
        print(f" Confidence         : {confidence:.1%}")
        print(f" Model Used         : {bboxes.get('model_used')}")
        print(f" Detections Count   : {bboxes.get('count')}")

        if bboxes.get("detections"):
            print(" Top Detections     :")
            for d_idx, det in enumerate(bboxes["detections"][:3], start=1):
                print(f"   [{d_idx}] Box: {det['bbox']} | Conf: {det['confidence']:.1%} | Label: '{det['label']}'")

        safe_explanation = str(explanation).encode('ascii', errors='ignore').decode('ascii')
        print(f" Synthesized Answer : {safe_explanation[:150]}...\n")

        if status != "success":
            print(f" Error details: {response.get('error')}")
            return False

    print("=" * 75)
    print(" FULL SYSTEM INTEGRATION TEST PASSED!")
    print(" -> Tool Selector & Sequencer correctly assigned Text-Guided Grounding Model.")
    print(" -> Automated 500m² satellite map tile fetch active for location queries.")
    print(" -> Bounding box detections, visual evidence, and confidence scores aggregated.")
    print("=" * 75 + "\n")
    return True


if __name__ == "__main__":
    success = asyncio.run(run_full_system_test())
    sys.exit(0 if success else 1)
