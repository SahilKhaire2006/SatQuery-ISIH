from typing import Dict, List
import asyncio
import numpy as np

from models.vqa_model import VQAModel
from models.grounding_model import GroundingModel
from models.change_detection_model import ChangeDetectionModel
from models.sar_fusion_model import SARFusionModel

try:
    from models.building_detector import BuildingDetector
except ImportError:
    BuildingDetector = GroundingModel

try:
    from models.roboflow_building_detector import RoboflowBuildingDetector
except ImportError:
    RoboflowBuildingDetector = None

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ExecutionEngine:
    """
    Executes selected specialist models and aggregates results
    """
    
    def __init__(self):
        # Try to load Roboflow detector first, fallback to U-Net
        try:
            roboflow_detector = RoboflowBuildingDetector()
            if roboflow_detector.loaded:
                logger.info("Using Roboflow Building Detector (primary)")
                building_detector = roboflow_detector
            else:
                logger.warning("Roboflow detector not available, falling back to U-Net")
                building_detector = BuildingDetector()
        except Exception as e:
            logger.warning(f"Failed to load Roboflow detector: {e}, falling back to U-Net")
            building_detector = BuildingDetector()
        
        self.models = {
            'vqa_model': VQAModel(),
            'grounding_model': GroundingModel(),
            'building_detector': building_detector,  # Roboflow or U-Net fallback
            'roboflow_building_detector': building_detector,  # Alias for explicit routing
            'change_detection_model': ChangeDetectionModel(),
            'sar_fusion_model': SARFusionModel()
        }
        logger.info("Execution Engine initialized with all models")
    
    async def execute(
        self,
        tools: List[Dict],
        image_data: np.ndarray,
        query: str,
        parameters: Dict
    ) -> Dict:
        """
        Execute selected tools in sequence
        """
        results = {}
        execution_log = []
        overall_confidence = []
        
        for tool in sorted(tools, key=lambda x: x['order']):
            tool_id = tool['tool_id']
            tool_params = tool['parameters']
            
            logger.info(f"Executing tool: {tool_id}")
            
            try:
                # Execute model
                model = self.models[tool_id]
                res = model.predict(
                    image=image_data,
                    query=query,
                    parameters=tool_params
                )
                if asyncio.iscoroutine(res):
                    result = await res
                else:
                    result = res
                
                # Store results
                results[tool_id] = result['output']
                overall_confidence.append(result['confidence'])
                
                execution_log.append({
                    'tool_id': tool_id,
                    'status': 'success',
                    'confidence': result['confidence'],
                    'execution_time': result.get('execution_time', 0)
                })
                
            except Exception as e:
                logger.error(f"Error executing {tool_id}: {str(e)}")
                execution_log.append({
                    'tool_id': tool_id,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Aggregate results
        aggregated_results = self._aggregate_results(results, tools)
        
        # Calculate confidence (convert numpy float to Python float for JSON serialization)
        confidence = float(np.mean(overall_confidence)) if overall_confidence else 0.0
        
        return {
            'results': aggregated_results,
            'confidence': confidence,
            'execution_log': execution_log
        }
    
    def _aggregate_results(self, results: Dict, tools: List[Dict]) -> Dict:
        """
        Aggregate results from multiple tools into coherent output
        """
        aggregated = {
            'answer': '',
            'visual_output': None,
            'confidence_scores': {},
            'details': {}
        }
        
        # Handle case where no tools were executed
        if not tools or not results:
            logger.warning("No results to aggregate")
            return aggregated
        
        # Primary answer from first tool
        primary_tool = tools[0]['tool_id']
        if primary_tool in results:
            primary_result = results[primary_tool]
            if isinstance(primary_result, dict):
                aggregated['answer'] = primary_result.get('answer', primary_result.get('query', ''))
                aggregated['visual_output'] = primary_result.get('visualization', None)
        
        # Add supporting information from other tools
        for tool_id, result in results.items():
            if isinstance(result, dict):
                aggregated['details'][tool_id] = result
                if 'confidence' in result:
                    aggregated['confidence_scores'][tool_id] = result['confidence']
        
        return aggregated
