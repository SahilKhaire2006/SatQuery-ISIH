import uuid
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from models.llm.groq_engine import GroqLLMEngine
from agentic_layer.input_validator import InputValidator
from agentic_layer.query_interpreter import QueryInterpreter
from agentic_layer.tool_selector import ToolSelector
from agentic_layer.execution_engine import ExecutionEngine
from agentic_layer.result_aggregator import ResultAggregator
from geospatial.metadata_parser import GeoMetadataParser
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AgenticOrchestrator:
    """
    Main orchestration layer coordinating Groq LLM reasoning, geospatial processing, and specialist tools
    """

    def __init__(self, api_key: Optional[str] = None):
        self.llm_engine = GroqLLMEngine(api_key=api_key)
        self.input_validator = InputValidator()
        self.query_interpreter = QueryInterpreter(llm_engine=self.llm_engine)
        self.tool_selector = ToolSelector(llm_engine=self.llm_engine)
        self.execution_engine = ExecutionEngine()
        self.result_aggregator = ResultAggregator(llm_engine=self.llm_engine)
        self.geo_parser = GeoMetadataParser()
        logger.info("Agentic Orchestrator with Groq LLM & Geospatial engine initialized")

    async def process_request(
        self,
        session_id: str,
        query: str,
        image_data: bytes,
        image_filename: str,
        geo_metadata: Optional[str] = None
    ) -> Dict:
        """
        Main processing pipeline for incoming requests
        """
        query_id = str(uuid.uuid4())
        logger.info(f"Processing query {query_id} for session {session_id}")

        try:
            # Step 1: Input Validation & Geospatial Metadata Parsing
            validation_result = await self.input_validator.validate(
                query=query,
                image_data=image_data,
                image_filename=image_filename,
                geo_metadata=geo_metadata
            )

            if not validation_result['valid']:
                return {
                    'query_id': query_id,
                    'status': 'failed',
                    'error': validation_result['errors'],
                    'results': {},
                    'confidence': 0.0,
                    'audit_log': {'validation': validation_result}
                }

            parsed_geo = self.geo_parser.parse_metadata(geo_metadata)

            # Step 2: Query Interpretation via Groq LLM & Coordinate Extraction
            interpretation = await self.query_interpreter.interpret(
                query=query,
                task_type=validation_result.get('task_type', 'general')
            )
            interpretation['geospatial_context'] = parsed_geo

            # Step 3: Tool Selection & Routing via Groq LLM (USP-1)
            selected_tools = await self.tool_selector.select_tools(
                interpretation=interpretation,
                task_type=interpretation['task_type']
            )

            # Step 4: Map Tile Acquisition & Execution Pipeline
            image_for_execution = validation_result['processed_image']
            image_filename = validation_result.get('image_filename', '')
            
            # Check if disaster grounding model is selected (Model 2)
            # Disaster model fetches its own real satellite imagery internally,
            # so we skip the default tile fetch to avoid unnecessary API calls.
            is_disaster_query = any(
                t.get('tool_id') == 'disaster_grounding_model'
                for t in selected_tools
            )
            
            if is_disaster_query:
                # Disaster model handles its own imagery — provide placeholder
                logger.info("Disaster query detected — Model 2 will fetch its own real satellite imagery")
                if image_for_execution is None:
                    image_for_execution = np.zeros((512, 512, 3), dtype=np.uint8)
            else:
                # Detect if uploaded image is dummy synthetic canvas (<= 150px) or missing
                is_synthetic_or_missing = (
                    image_for_execution is None
                    or (isinstance(image_for_execution, np.ndarray) and (image_for_execution.shape[0] <= 150 or image_for_execution.shape[1] <= 150))
                    or 'synthetic' in str(image_filename).lower()
                )
                
                # Only fetch a map tile when the user did NOT upload a real image.
                # When a real image is uploaded, always use it for analysis.
                if is_synthetic_or_missing:
                    try:
                        from geospatial.map_fetcher import geocode_location, fetch_satellite_image_tile
                        lat, lon, display_name = geocode_location(query)
                        tile_info = fetch_satellite_image_tile(lat, lon, area_meters=500.0)
                        image_for_execution = tile_info['image']
                        
                        # Update geospatial metadata with real geocoded coordinates
                        parsed_geo['center_coords'] = {'lat': lat, 'lon': lon}
                        parsed_geo['spatial_bounds'] = tile_info.get('bbox_geo', [lon-0.002, lat-0.002, lon+0.002, lat+0.002])
                        parsed_geo['display_name'] = display_name
                        interpretation['geospatial_context'] = parsed_geo
                        
                        logger.info(f"Automated 500m satellite tile successfully fetched for '{display_name}' ({lat:.5f}, {lon:.5f})")
                    except Exception as map_err:
                        logger.warning(f"Could not fetch map tile ({map_err}). Using default image.")
                        if image_for_execution is None:
                            from PIL import Image
                            if Path("satelite-img.png").exists():
                                image_for_execution = np.array(Image.open("satelite-img.png").convert("RGB"))
                            else:
                                image_for_execution = np.zeros((512, 512, 3), dtype=np.uint8)
                else:
                    logger.info(f"Using user-uploaded image for analysis (shape: {image_for_execution.shape})")



            execution_result = await self.execution_engine.execute(
                tools=selected_tools,
                image_data=image_for_execution,
                query=query,
                parameters=interpretation['parameters']
            )

            # Step 5: Result Aggregation & Visual Evidence Grounding (USP-2)
            aggregated = await self.result_aggregator.aggregate(
                query=query,
                interpretation=interpretation,
                tool_results=execution_result['results'],
                image=image_for_execution
            )

            # Combine final results
            final_results = execution_result['results']
            final_results['aggregated_summary'] = aggregated['summary']
            final_results['evidence'] = aggregated['evidence']
            final_results['visual_evidence'] = aggregated.get('visual_evidence', {})
            final_results['geospatial_metadata'] = parsed_geo
            
            tool_details = execution_result['results'].get('details', execution_result['results'])
            
            def _extract_dets(obj):
                dets = []
                if isinstance(obj, dict):
                    if 'detections' in obj and isinstance(obj['detections'], list):
                        dets.extend(obj['detections'])
                    for k, v in obj.items():
                        if isinstance(v, dict):
                            dets.extend(_extract_dets(v))
                return dets

            # Combine detections from ALL executed tools (buildings, water bodies, vegetation, etc.)
            all_detections = []
            models_used = []

            for tool_k, tool_v in tool_details.items():
                found_dets = _extract_dets(tool_v)
                if found_dets:
                    all_detections.extend(found_dets)
                    models_used.append(tool_k)

            model_used_name = ", ".join(models_used) if models_used else "none"
            img_shape = image_for_execution.shape if image_for_execution is not None else (0, 0)
            
            final_results['bounding_boxes'] = {
                'detections': [
                    {
                        'label': d.get('label', 'object'),
                        'confidence': float(d.get('confidence', 0.85)),
                        'bbox': d.get('bbox', [0, 0, 0, 0])  # [x1, y1, x2, y2]
                    }
                    for d in all_detections
                ],
                'status': 'success' if all_detections else 'completed',
                'image_dimensions': {
                    'width': img_shape[1] if len(img_shape) > 1 else 0,
                    'height': img_shape[0] if len(img_shape) > 0 else 0
                },
                'count': len(all_detections),
                'model_used': model_used_name
            }
            
            logger.info(f"Bounding boxes: {len(all_detections)} detections from {final_results['bounding_boxes']['model_used']}, visual_evidence keys: {list(final_results['visual_evidence'].keys())}")

            from utils.json_sanitizer import sanitize_for_json
            response_dict = {
                'query_id': query_id,
                'status': 'success',
                'explanation': aggregated['summary'],
                'results': final_results,
                'confidence': execution_result['confidence'],
                'audit_log': {
                    'validation': {
                        'valid': validation_result['valid'],
                        'errors': validation_result['errors'],
                        'warnings': validation_result['warnings'],
                        'image_shape': validation_result['image_shape'],
                        'task_type': validation_result['task_type']
                    },
                    'geospatial': parsed_geo,
                    'interpretation': interpretation,
                    'selected_tools': selected_tools,
                    'execution': execution_result['execution_log'],
                    'evidence_summary': aggregated['evidence']
                },
                'state': execution_result.get('state', {})
            }
            return sanitize_for_json(response_dict)

        except Exception as e:
            logger.error(f"Error processing query {query_id}: {str(e)}")
            from utils.json_sanitizer import sanitize_for_json
            return sanitize_for_json({
                'query_id': query_id,
                'status': 'error',
                'error': str(e),
                'results': {},
                'confidence': 0.0,
                'audit_log': {}
            })

    def get_available_tools(self) -> List[str]:
        """Return list of available tools"""
        return self.tool_selector.get_tool_registry()
