import sys
import os
import asyncio
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geospatial.metadata_parser import GeoMetadataParser
from geospatial.coordinate_system import CoordinateTransformer
from geospatial.spatial_queries import SpatialQueryEngine
from geospatial.tile_processor import TileProcessor
from agentic_layer.query_interpreter import QueryInterpreter
from agentic_layer.orchestrator import AgenticOrchestrator


def test_geo_metadata_parser():
    parser = GeoMetadataParser()
    meta_json = '{"crs": "EPSG:4326", "bounds": [77.10, 28.50, 77.30, 28.70], "resolution_m": 5.0}'
    parsed = parser.parse_metadata(meta_json)
    assert parsed['has_geospatial'] is True
    assert parsed['crs'] == 'EPSG:4326'
    assert parsed['resolution_m'] == 5.0
    assert parsed['center_lat'] == 28.6000


def test_coordinate_transformer():
    transformer = CoordinateTransformer()
    bounds = [77.0, 28.0, 78.0, 29.0]
    lon, lat = transformer.pixel_to_latlon(50, 50, 100, 100, bounds)
    assert 77.0 <= lon <= 78.0
    assert 28.0 <= lat <= 29.0

    px, py = transformer.latlon_to_pixel(lon, lat, 100, 100, bounds)
    assert abs(px - 50) <= 2
    assert abs(py - 50) <= 2

    x_3857, y_3857 = transformer.epsg4326_to_epsg3857(77.2090, 28.6139)
    assert x_3857 > 0
    assert y_3857 > 0


def test_spatial_query_engine():
    engine = SpatialQueryEngine()
    # Distance between New Delhi (28.6139, 77.2090) and Connaught Place (28.6315, 77.2167)
    dist = engine.haversine_distance_m(28.6139, 77.2090, 28.6315, 77.2167)
    assert 1500 <= dist <= 2500

    res = engine.evaluate_spatial_filter(28.6139, 77.2090, 28.6315, 77.2167, radius_m=3000)
    assert res['matched'] is True


def test_tile_processor():
    processor = TileProcessor(tile_size=64, overlap=16)
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    tiles = processor.generate_tiles(img)
    assert len(tiles) >= 4

    merged = processor.merge_tiled_detections([
        {'x_offset': 10, 'y_offset': 20, 'detection': {'bbox': [5, 5, 25, 25]}}
    ])
    assert merged[0]['bbox'] == [15, 25, 35, 45]


def test_query_interpreter_coordinate_extraction():
    async def _run():
        interpreter = QueryInterpreter()
        query = "Locate building at 28.6139, 77.2090"
        res = await interpreter.interpret(query)
        assert 'spatial_metadata' in res
        assert 'coordinates' in res['spatial_metadata']
        assert res['spatial_metadata']['coordinates'] == [28.6139, 77.2090]

    asyncio.run(_run())


def test_orchestrator_phase6_integration():
    async def _run():
        orchestrator = AgenticOrchestrator()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', dummy_img)
        image_bytes = buffer.tobytes()

        geo_meta = '{"crs": "EPSG:4326", "bounds": [77.10, 28.50, 77.30, 28.70]}'

        resp = await orchestrator.process_request(
            session_id="sess_phase6_1",
            query="Analyze satellite image at 28.6139, 77.2090",
            image_data=image_bytes,
            image_filename="test.tif",
            geo_metadata=geo_meta
        )
        assert resp['status'] == 'success'
        assert 'geospatial' in resp['audit_log']
        assert resp['audit_log']['geospatial']['has_geospatial'] is True

    asyncio.run(_run())


if __name__ == '__main__':
    test_geo_metadata_parser()
    test_coordinate_transformer()
    test_spatial_query_engine()
    test_tile_processor()
    test_query_interpreter_coordinate_extraction()
    test_orchestrator_phase6_integration()
    print("All Phase 6 Geospatial Processing tests passed successfully!")
