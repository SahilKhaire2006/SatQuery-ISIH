from typing import Dict, List

from config.settings import AVAILABLE_TOOLS
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ToolSelector:
    """
    Selects and sequences specialist models based on task requirements
    """
    
    def __init__(self):
        self.tool_registry = {
            'vqa_model': {
                'name': 'VQA Model',
                'tasks': ['vqa', 'general_query', 'description'],
                'priority': 1
            },
            'grounding_model': {
                'name': 'Grounding Model',
                'tasks': ['grounding', 'localization', 'detection'],
                'priority': 2
            },
            'change_detection_model': {
                'name': 'Change Detection Model',
                'tasks': ['change_detection', 'comparison', 'temporal'],
                'priority': 3
            },
            'sar_fusion_model': {
                'name': 'SAR Fusion Model',
                'tasks': ['sar_fusion', 'cross_modal', 'radar'],
                'priority': 4
            }
        }
    
    async def select_tools(self, interpretation: Dict, task_type: str) -> List[Dict]:
        """
        Select appropriate tools based on interpretation
        """
        selected = []
        intent = interpretation.get('intent', 'general_query')
        
        # Primary tool selection based on task type
        primary_tool = self._select_primary_tool(task_type)
        if primary_tool:
            selected.append({
                'tool_id': primary_tool,
                'tool_name': self.tool_registry[primary_tool]['name'],
                'order': 1,
                'parameters': self._get_tool_parameters(primary_tool, interpretation)
            })
        
        # Secondary tools based on intent and entities
        secondary_tools = self._select_secondary_tools(interpretation, primary_tool)
        for idx, tool in enumerate(secondary_tools, start=2):
            selected.append({
                'tool_id': tool,
                'tool_name': self.tool_registry[tool]['name'],
                'order': idx,
                'parameters': self._get_tool_parameters(tool, interpretation)
            })
        
        logger.info(f"Selected {len(selected)} tools: {[t['tool_id'] for t in selected]}")
        return selected
    
    def _select_primary_tool(self, task_type: str) -> str:
        """Select primary tool based on task type"""
        task_to_tool = {
            'vqa': 'vqa_model',
            'grounding': 'grounding_model',
            'change_detection': 'change_detection_model',
            'sar_fusion': 'sar_fusion_model'
        }
        return task_to_tool.get(task_type, 'vqa_model')
    
    def _select_secondary_tools(self, interpretation: Dict, primary_tool: str) -> List[str]:
        """Select secondary supporting tools"""
        secondary = []
        intent = interpretation.get('intent', '')
        
        # If localization is needed and primary isn't grounding, add it
        if intent == 'localization' and primary_tool != 'grounding_model':
            secondary.append('grounding_model')
        
        # If SAR mentioned in parameters, add fusion model
        params = interpretation.get('parameters', {})
        if 'sar' in params.get('modalities', []) and primary_tool != 'sar_fusion_model':
            secondary.append('sar_fusion_model')
        
        return secondary
    
    def _get_tool_parameters(self, tool_id: str, interpretation: Dict) -> Dict:
        """Extract relevant parameters for specific tool"""
        params = interpretation.get('parameters', {})
        
        if tool_id == 'grounding_model':
            return {
                'target_object': params.get('target_object', ''),
                'entities': interpretation.get('entities', [])
            }
        elif tool_id == 'change_detection_model':
            return {
                'temporal': params.get('temporal', False),
                'comparison_type': params.get('comparison_type', 'general')
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
