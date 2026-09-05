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
        Interpret query using Groq LLM reasoning to classify intent
        Returns structured interpretation with explicit intent classification
        """
        logger.info(f"Interpreting query: '{query}' (task_hint: {task_type})")

        # Build context with task hint if provided
        context = {'task_hint': task_type} if task_type else None

        # Get structured interpretation from LLM engine with intent classification
        interpretation = await self.llm_engine.interpret_query(query, context=context)

        # Classify intent based on LLM output and query keywords
        intent = self._classify_intent(query, interpretation)
        interpretation['intent'] = intent
        interpretation['intent_confidence'] = interpretation.get('confidence', 0.85)

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

        logger.info(f"Query classified as intent: {intent} (task_type: {interpretation.get('task_type')})")
        return interpretation
    
    def _classify_intent(self, query: str, interpretation: Dict) -> str:
        """
        Classify query intent into specific categories for tool routing
        Priority order: disaster > building_detection > water_detection > vegetation_detection > change_detection > general_vqa
        """
        query_lower = query.lower()
        task_type = interpretation.get('task_type', 'vqa')
        entities = interpretation.get('entities', [])
        
        # Disaster intent (highest priority — Model 2)
        flood_keywords = ['flood', 'inundation', 'water level', 'submerged',
                          'flood progression', 'flood extent', 'riverbank overflow',
                          'deluge', 'waterlogging', 'flash flood']
        earthquake_keywords = ['earthquake', 'seismic', 'structural damage',
                               'collapsed building', 'richter', 'magnitude',
                               'tremor', 'aftershock', 'quake']
        disaster_keywords = ['disaster', 'evacuation', 'tsunami', 'cyclone',
                             'hurricane', 'landslide', 'wildfire', 'storm surge',
                             'rescue', 'damage assessment', 'relief',
                             'emergency response', 'catastrophe']

        if any(kw in query_lower for kw in flood_keywords):
            return 'disaster_flood'
        if any(kw in query_lower for kw in earthquake_keywords):
            return 'disaster_earthquake'
        if any(kw in query_lower for kw in disaster_keywords):
            return 'disaster_general'

        # Building detection intent
        building_keywords = ['building', 'structure', 'warehouse', 'factory', 'house', 'construction', 'rooftop']
        building_action_keywords = ['count', 'how many', 'detect', 'locate', 'find', 'identify', 'show', 'where', 'any', 'are there', 'is there']
        if any(kw in query_lower for kw in building_keywords):
            # Trigger building detection if any action keyword is present, OR if the query is short/simple
            if any(action in query_lower for action in building_action_keywords) or len(query_lower.split()) <= 8:
                return 'building_detection'
        
        # Water detection intent
        water_keywords = ['water', 'river', 'lake', 'pond', 'ocean', 'sea', 'reservoir', 'stream']
        if any(kw in query_lower for kw in water_keywords):
            return 'water_detection'
        
        # Vegetation detection intent
        vegetation_keywords = ['vegetation', 'forest', 'tree', 'plant', 'crop', 'green', 'ndvi', 'greenery']
        if any(kw in query_lower for kw in vegetation_keywords):
            return 'vegetation_detection'
        
        # Change detection intent
        change_keywords = ['change', 'difference', 'compare', 'before', 'after', 'temporal', 'evolve']
        if any(kw in query_lower for kw in change_keywords):
            return 'change_detection'
        
        # Default to general VQA
        return 'general_vqa'

    def _extract_coordinates(self, query: str) -> Optional[list]:
        """Extract latitude and longitude floating point numbers from query text"""
        from geospatial.map_fetcher import extract_coordinates_from_text
        parsed = extract_coordinates_from_text(query)
        if parsed is not None:
            return [parsed[0], parsed[1]]
        return None

