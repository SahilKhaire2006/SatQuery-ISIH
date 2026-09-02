"""
Manual Testing Script for Text-Guided Grounding Specialist Model

Supports passing:
1. Coordinates or location/address (e.g. --location "28.6139, 77.2090" or --location "Taj Mahal, Agra")
   -> Automatically geocodes location and downloads a high-res 500m x 500m satellite image tile!
2. Area size in meters (e.g. --area_meters 500)
3. Custom local image file (e.g. --image satelite-img.png)
4. Target query (e.g. --query "locate the main building structure")

Usage Examples:
    python test_manual_grounding.py --location "Taj Mahal, Agra" --query "locate the main dome structure"
    python test_manual_grounding.py --location "28.6139, 77.2090" --area_meters 500 --query "find water body reservoir"
    python test_manual_grounding.py --image "satelite-img.png" --query "a commercial building complex"
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image

from geospatial.map_fetcher import geocode_location, fetch_satellite_image_tile
from text_guided_grounding.inference import ground
from text_guided_grounding.visualizer import visualize_grounding


def main():
    parser = argparse.ArgumentParser(
        description="Manual Test CLI for Text-Guided Grounding (Address / Coordinates / 500m² Map Area Support)"
    )
    parser.add_argument("--location", type=str, default=None, help="Coordinates (lat, lon) or address/landmark name (e.g., 'Taj Mahal, Agra' or '28.6139, 77.2090')")
    parser.add_argument("--area_meters", type=float, default=500.0, help="Square tile coverage side length in meters (default: 500m)")
    parser.add_argument("--image", type=str, default="satelite-img.png", help="Path to local input satellite image file (used if --location is not set)")
    parser.add_argument("--query", type=str, default="locate the main building facility", help="Natural language referring expression query")
    parser.add_argument("--out", type=str, default="manual_test_result.jpg", help="Path to save output visualization overlay")

    args = parser.parse_args()

    print("=" * 70)
    print(f"      SATQUERY — MANUAL TEXT-GUIDED GROUNDING ({int(args.area_meters)}m x {int(args.area_meters)}m MAP MODE)")
    print("=" * 70)

    # 1. Acquire Image: Fetch Satellite Map Tile if --location provided
    if args.location:
        print(f" Step 1: Resolving location / address: '{args.location}'...")
        try:
            lat, lon, display_name = geocode_location(args.location)
            print(f" -> Geocoded Location : {display_name}")
            print(f" -> Center Coordinates: ({lat:.5f}, {lon:.5f})")

            print(f"\n Step 2: Fetching high-resolution {int(args.area_meters)}m x {int(args.area_meters)}m satellite map tile...")
            map_info = fetch_satellite_image_tile(lat, lon, area_meters=args.area_meters)
            img_input = map_info["image_path"]
            print(f" -> Downloaded Satellite Tile: {img_input}")

        except Exception as e:
            print(f"[Location Error] {e}")
            sys.exit(1)
    else:
        img_input = Path(args.image)
        if not img_input.exists():
            print(f"[Error] Local satellite image file '{img_input}' not found!")
            print("Please provide a location via `--location \"address or lat,lon\"` or a local image path via `--image <path>`.")
            sys.exit(1)
        print(f" Using Local Satellite Image: {img_input}")

    print(f"\n Step 3: Running neural text-guided visual grounding...")
    print(f" Query Text: '{args.query}'")

    try:
        # Execute Text-Guided Visual Grounding Model
        result = ground(image=img_input, query=args.query, top_k=5)
        top_bbox = result["bbox"]
        confidence = result["confidence"]
        candidates = result["candidates"]

        print("\n Inference Results:")
        print(f" -> Top Predicted Box [xmin, ymin, xmax, ymax] : {top_bbox}")
        print(f" -> Calibrated Confidence Score             : {confidence:.1%}")
        print(f" -> Region Candidates Returned              : {len(candidates)}")

        if candidates:
            print("\n Top Region Candidates:")
            for idx, c in enumerate(candidates, start=1):
                print(f"   [{idx}] Box: {c['bbox']} | Conf: {c['confidence']:.1%} | Label: '{c['label']}'")

        # Render visual evidence overlay
        out_path = Path(args.out)
        visualize_grounding(
            image=img_input,
            pred_bbox=top_bbox,
            confidence=confidence,
            query=args.query,
            output_path=out_path
        )

        print("\n [Success] Rendered bounding box overlay saved to:")
        print(f"  -> {out_path.resolve()}\n")
        print("=" * 70)

    except Exception as e:
        print(f"\n[Execution Failure] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
