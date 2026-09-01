import uuid
from typing import Dict, List, Optional
from datetime import datetime

from agentic_layer.input_validator import InputValidator
from agentic_layer.query_interpreter import QueryInterpreter
from agentic_layer.tool_selector import ToolSelector
from agentic_layer.execution_engine import ExecutionEngine
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AgenticOrchestrator:
    """
    Main orchestration layer coordinating all agentic components
    """
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.query_interpreter = QueryInterpreter()
        self.tool_selector = ToolSelector()
        self.execution_engine = ExecutionEngine()
        logger.info("Agentic Orchestrator initialized")
    
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
            # Step 1: Input Validation
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
            
            # Step 2: Query Interpretation
            interpretation = await self.query_interpreter.interpret(
                query=query,
                task_type=validation_result.get('task_type', 'general')
            )
            
            # Step 3: Tool Selection
            selected_tools = await self.tool_selector.select_tools(
                interpretation=interpretation,
                task_type=interpretation['task_type']
            )
            
            # Step 4: Execution
            execution_result = await self.execution_engine.execute(
                tools=selected_tools,
                image_data=validation_result['processed_image'],
                query=query,
                parameters=interpretation['parameters']
            )
            
            # Compile results
            return {
                'query_id': query_id,
                'status': 'success',
                'results': execution_result['results'],
                'confidence': execution_result['confidence'],
                'audit_log': {
                    'validation': validation_result,
                    'interpretation': interpretation,
                    'selected_tools': selected_tools,
                    'execution': execution_result['execution_log']
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
