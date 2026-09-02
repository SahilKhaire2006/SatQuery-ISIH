import re
from typing import Dict, Optional
from models.llm.groq_engine import GroqLLMEngine
from utils.logger import setup_logger

logger = setup_logger(__name__)


class QueryInterpreter:
    """
    LLM-powered query interpreter with geographic coordinate & spatial reference extraction
    """

    def __init__(self, llm_engine: Optional[GroqLLMEngine] = None):
        self.llm_engine = llm_engine or GroqLLMEngine()

    async def interpret(self, query: str, task_type: Optional[str] = None) -> Dict:
        """
        Interpret query using Groq LLM reasoning (or fallback)
        """
        logger.info(f"Interpreting query: '{query}' (task_hint: {task_type})")

        # Build context with task hint if provided
        context = {'task_hint': task_type} if task_type else None

        # Get structured interpretation from LLM engine
        interpretation = await self.llm_engine.interpret_query(query, context=context)

        # Ensure standard structure expected downstream
        if 'parameters' not in interpretation:
            interpretation['parameters'] = {}

        target_obj = interpretation.get('target_object', '')
        if target_obj and 'target_object' not in interpretation['parameters']:
            interpretation['parameters']['target_object'] = target_obj

        modalities = interpretation.get('modalities', ['optical'])
        if 'modalities' not in interpretation['parameters']:
            interpretation['parameters']['modalities'] = modalities

        # Extract Geographic Coordinates regex (e.g. 28.6139°N, 77.2090°E or 28.6139, 77.2090)
        coords = self._extract_coordinates(query)
        if 'spatial_metadata' not in interpretation or not interpretation['spatial_metadata']:
            interpretation['spatial_metadata'] = {}

        if coords:
            interpretation['spatial_metadata']['coordinates'] = coords
            interpretation['parameters']['coordinates'] = coords

        logger.info(f"Query interpreted as task_type: {interpretation.get('task_type')}")
        return interpretation

    def _extract_coordinates(self, query: str) -> Optional[list]:
        """Extract latitude and longitude floating point numbers from query text"""
        # Match formats like 28.6139 N, 77.2090 E or 28.6139, 77.2090
        coord_pattern = r'(-?\d+\.\d+)\s*(?:°?\s*[NSns])?\s*,\s*(-?\d+\.\d+)\s*(?:°?\s*[EWew])?'
        match = re.search(coord_pattern, query)
        if match:
            try:
                lat = float(match.group(1))
                lon = float(match.group(2))
                return [lat, lon]
            except ValueError:
                pass
        return None
