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
        Select and sequence tools using Groq LLM reasoning (or fallback)
        """
        available = self.get_tool_registry()

        # Execute LLM tool selection
        tools = await self.llm_engine.select_tools(interpretation, available)

        q_lower = interpretation.get('original_query', '').lower()
        intent = interpretation.get('intent', '')
        
        # Automatic multi-tool selection for counting and localization queries
        is_counting_query = any(k in q_lower for k in ['count', 'how many', 'number of'])
        if is_counting_query:
            has_grounding = any(t.get('tool_id') == 'grounding_model' for t in tools)
            if not has_grounding:
                tools.append({
                    'tool_id': 'grounding_model',
                    'tool_name': 'Grounding Model',
                    'order': len(tools) + 1,
                    'parameters': {
                        'target_object': interpretation.get('target_object', 'building'),
                        'entities': interpretation.get('entities', ['building'])
                    },
                    'rationale': 'Automatic grounding execution for counting & object localization'
                })

        # Enforce USP-1 Directive (Optical-SAR Fusion Routing):
        # If task is SAR fusion or SAR is present in modalities, ensure sar_fusion_model is primary (order 1)
        # and bypass stand-alone optical grounding unless explicitly required.
        params = interpretation.get('parameters', {})
        modalities = params.get('modalities', interpretation.get('modalities', []))
        
        is_sar_query = (
            task_type == 'sar_fusion' or
            'sar' in modalities or
            'radar' in interpretation.get('original_query', '').lower()
        )

        if is_sar_query:
            logger.info("USP-1 Directive Triggered: SAR query detected. Routing directly to SAR Fusion Model.")
            # Re-order tool list so sar_fusion_model is primary
            sar_tool_present = any(t.get('tool_id') == 'sar_fusion_model' for t in tools)
            if not sar_tool_present:
                tools.insert(0, {
                    'tool_id': 'sar_fusion_model',
                    'tool_name': 'SAR Fusion Model',
                    'order': 1,
                    'parameters': {'modalities': ['optical', 'sar']},
                    'rationale': 'USP-1 SAR Fusion routing'
                })
            else:
                # Make sar_fusion_model the first tool
                tools = sorted(tools, key=lambda t: 0 if t.get('tool_id') == 'sar_fusion_model' else t.get('order', 2))

            # Filter out grounding_model if present to enforce USP-1 bypass unless requested specifically
            if interpretation.get('intent') != 'localization':
                tools = [t for t in tools if t.get('tool_id') != 'grounding_model']

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

        if tool_id == 'grounding_model':
            return {
                'target_object': params.get('target_object', interpretation.get('target_object', '')),
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
