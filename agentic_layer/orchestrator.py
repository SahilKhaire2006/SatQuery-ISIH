import uuid
from typing import Dict, List, Optional

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

            # Step 4: Execution Pipeline
            execution_result = await self.execution_engine.execute(
                tools=selected_tools,
                image_data=validation_result['processed_image'],
                query=query,
                parameters=interpretation['parameters']
            )

            # Step 5: Result Aggregation & Visual Evidence Grounding (USP-2)
            aggregated = await self.result_aggregator.aggregate(
                query=query,
                interpretation=interpretation,
                tool_results=execution_result['results'],
                image=validation_result['processed_image']
            )

            # Combine final results
            final_results = execution_result['results']
            final_results['aggregated_summary'] = aggregated['summary']
            final_results['evidence'] = aggregated['evidence']
            final_results['visual_evidence'] = aggregated.get('visual_evidence', {})
            final_results['geospatial_metadata'] = parsed_geo
            
            # Always include bounding boxes structure for frontend
            grounding_output = execution_result['results'].get('grounding_model', {})
            building_output = execution_result['results'].get('building_detector', {})
            
            # Prioritize building detector results if available
            if building_output and isinstance(building_output, dict):
                detection_source = building_output.get('output', building_output)
            elif grounding_output and isinstance(grounding_output, dict):
                detection_source = grounding_output.get('output', grounding_output)
            else:
                detection_source = {}
            
            detections = detection_source.get('detections', []) if isinstance(detection_source, dict) else []
            img_shape = validation_result['image_shape']
            
            final_results['bounding_boxes'] = {
                'detections': [
                    {
                        'label': d.get('label', 'object'),
                        'confidence': d.get('confidence', 0.0),
                        'bbox': d.get('bbox', [0, 0, 0, 0])  # [x1, y1, x2, y2]
                    }
                    for d in detections
                ],
                'status': detection_source.get('status', 'not_executed') if isinstance(detection_source, dict) else 'not_executed',
                'image_dimensions': {
                    'width': img_shape[1] if len(img_shape) > 1 else 0,
                    'height': img_shape[0] if len(img_shape) > 0 else 0
                },
                'count': len(detections),
                'model_used': 'building_detector' if building_output else 'grounding_model' if grounding_output else 'none'
            }
            
            logger.info(f"Bounding boxes: {len(detections)} detections from {final_results['bounding_boxes']['model_used']}, visual_evidence keys: {list(final_results['visual_evidence'].keys())}")

            return {
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

        except Exception as e:
            logger.error(f"Error processing query {query_id}: {str(e)}")
            return {
                'query_id': query_id,
                'status': 'error',
                'error': str(e),
                'results': {},
                'confidence': 0.0,
                'audit_log': {}
            }

    def get_available_tools(self) -> List[str]:
        """Return list of available tools"""
        return self.tool_selector.get_tool_registry()
