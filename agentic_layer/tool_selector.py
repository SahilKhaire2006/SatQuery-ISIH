from typing import Dict, List, Optional
from models.llm.groq_engine import GroqLLMEngine
from config.settings import AVAILABLE_TOOLS
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ToolSelector:
    """
    LLM-powered Tool Selector (with USP-1 SAR routing directive)
    """

    def __init__(self, llm_engine: Optional[GroqLLMEngine] = None):
        self.llm_engine = llm_engine or GroqLLMEngine()
        self.tool_registry = {
            'vqa_model': {
                'name': 'VQA Model',
                'tasks': ['vqa', 'general_query', 'description'],
                'priority': 3
            },
            'grounding_model': {
                'name': 'Grounding Model',
                'tasks': ['grounding', 'localization', 'detection'],
                'priority': 4
            },
            'building_detector': {
                'name': 'Building Detector (U-Net Fallback)',
                'tasks': ['building_detection', 'building_counting', 'structure_detection'],
                'priority': 2
            },
            'roboflow_building_detector': {
                'name': 'Roboflow Building Detector',
                'tasks': ['building_detection', 'building_counting', 'structure_detection'],
                'priority': 1  # Highest priority - use first
            },
            'waterbody_detector': {
                'name': 'Water Body Detector',
                'tasks': ['water_detection', 'waterbody_detection', 'water_counting', 'lake_detection', 'river_detection'],
                'priority': 1  # Highest priority - use first
            },
            'roboflow_waterbody_detector': {
                'name': 'Roboflow Water Body Detector',
                'tasks': ['water_detection', 'waterbody_detection', 'water_counting', 'lake_detection', 'river_detection'],
                'priority': 1  # Highest priority - use first
            },
            'spectral_index_model': {
                'name': 'Spectral Index Model',
                'tasks': ['vegetation_detection', 'ndvi', 'ndwi', 'spectral_analysis'],
                'priority': 2  # Lower priority than Roboflow water detector
            },
            'change_detection_model': {
                'name': 'Change Detection Model',
                'tasks': ['change_detection', 'comparison', 'temporal'],
                'priority': 2
            },
            'sar_fusion_model': {
                'name': 'SAR Fusion Model',
                'tasks': ['sar_fusion', 'cross_modal', 'radar'],
                'priority': 4
            }
        }

    async def select_tools(self, interpretation: Dict, task_type: str) -> List[Dict]:
        """
        Select and sequence tools using intent-based routing
        Uses LLM classification output (interpretation['intent']) for routing decisions
        """
        available = self.get_tool_registry()
        intent = interpretation.get('intent', 'general_vqa')
        q_lower = interpretation.get('original_query', '').lower()
        
        logger.info(f"Routing query with intent: {intent}")

        # Intent-based tool routing (explicit mapping)
        if intent == 'building_detection':
            logger.info("Intent routing: building_detection -> roboflow_building_detector")
            tools = [{
                'tool_id': 'roboflow_building_detector',
                'tool_name': 'Roboflow Building Detector',
                'order': 1,
                'parameters': {
                    'target_object': 'building',
                    'entities': interpretation.get('entities', ['building'])
                },
                'rationale': 'Intent-based routing for building detection using Roboflow API'
            }]
        
        elif intent == 'water_detection':
            logger.info("Intent routing: water_detection -> roboflow_waterbody_detector")
            tools = [{
                'tool_id': 'roboflow_waterbody_detector',
                'tool_name': 'Roboflow Water Body Detector',
                'order': 1,
                'parameters': {
                    'target_object': 'water',
                    'entities': interpretation.get('entities', ['water', 'lake', 'river'])
                },
                'rationale': 'Intent-based routing for water body detection using Roboflow API'
            }]
        
        elif intent == 'vegetation_detection':
            logger.info("Intent routing: vegetation_detection -> spectral_index_model (NDVI)")
            tools = [{
                'tool_id': 'spectral_index_model',
                'tool_name': 'Spectral Index Model (Vegetation)',
                'order': 1,
                'parameters': {
                    'index_type': 'ndvi',
                    'target_object': 'vegetation',
                    'entities': interpretation.get('entities', ['vegetation'])
                },
                'rationale': 'Intent-based routing for vegetation detection'
            }]
        
        elif intent == 'change_detection':
            logger.info("Intent routing: change_detection -> change_detection_model")
            tools = [{
                'tool_id': 'change_detection_model',
                'tool_name': 'Change Detection Model',
                'order': 1,
                'parameters': interpretation.get('parameters', {}),
                'rationale': 'Intent-based routing for temporal change analysis'
            }]
        
        else:
            # general_vqa or unknown intent - use LLM tool selection
            logger.info(f"Intent routing: {intent} -> LLM tool selection")
            tools = await self.llm_engine.select_tools(interpretation, available)

        # Re-assign sequential order & complement missing parameters
        for idx, tool in enumerate(tools, start=1):
            tool['order'] = idx
            if 'parameters' not in tool or not tool['parameters']:
                tool['parameters'] = self._get_tool_parameters(tool.get('tool_id', ''), interpretation)
            if 'tool_name' not in tool:
                tool['tool_name'] = self.tool_registry.get(tool.get('tool_id'), {}).get('name', tool.get('tool_id'))

        logger.info(f"Final tool sequence: {[t['tool_id'] for t in tools]}")
        return tools

    def _get_tool_parameters(self, tool_id: str, interpretation: Dict) -> Dict:
        """Extract relevant parameters for specific tool"""
        params = interpretation.get('parameters', {})

        if tool_id in ['grounding_model', 'building_detector', 'roboflow_building_detector']:
            return {
                'target_object': params.get('target_object', interpretation.get('target_object', 'building')),
                'entities': interpretation.get('entities', [])
            }
        elif tool_id in ['waterbody_detector', 'roboflow_waterbody_detector']:
            return {
                'target_object': params.get('target_object', interpretation.get('target_object', 'water')),
                'entities': interpretation.get('entities', ['water'])
            }
        elif tool_id == 'spectral_index_model':
            return {
                'index_type': params.get('index_type', 'ndvi'),
                'target_object': params.get('target_object', ''),
                'entities': interpretation.get('entities', [])
            }
        elif tool_id == 'change_detection_model':
            return {
                'temporal': params.get('temporal', True),
                'comparison_type': params.get('comparison_type', 'before_after')
            }
        elif tool_id == 'sar_fusion_model':
            return {
                'modalities': params.get('modalities', ['optical', 'sar'])
            }
        else:  # vqa_model
            return {
                'query': interpretation.get('original_query', '')
            }

    def get_tool_registry(self) -> List[str]:
        """Return list of available tools"""
        return list(self.tool_registry.keys())
