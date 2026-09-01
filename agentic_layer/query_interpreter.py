from typing import Dict
import re

from utils.logger import setup_logger

logger = setup_logger(__name__)


class QueryInterpreter:
    """
    Interprets natural language queries and classifies into task types
    """
    
    def __init__(self):
        self.task_patterns = {
            'vqa': [
                r'what\s+is',
                r'how\s+many',
                r'describe',
                r'explain',
                r'count'
            ],
            'grounding': [
                r'where\s+is',
                r'locate',
                r'find',
                r'detect',
                r'identify\s+location'
            ],
            'change_detection': [
                r'change',
                r'difference',
                r'compare',
                r'before\s+and\s+after',
                r'temporal'
            ],
            'sar_fusion': [
                r'sar',
                r'radar',
                r'fusion',
                r'optical.*sar',
                r'cross.*modal'
            ]
        }
    
    async def interpret(self, query: str, task_type: str = None) -> Dict:
        """
        Interpret query and extract parameters
        """
        query_lower = query.lower()
        
        # Determine task type if not provided
        if not task_type:
            task_type = self._classify_task_type(query_lower)
        
        # Extract parameters based on task type
        parameters = self._extract_parameters(query, task_type)
        
        # Generate structured representation
        structured_query = {
            'task_type': task_type,
            'original_query': query,
            'parameters': parameters,
            'intent': self._extract_intent(query, task_type),
            'entities': self._extract_entities(query)
        }
        
        logger.info(f"Query interpreted as task type: {task_type}")
        return structured_query
    
    def _classify_task_type(self, query: str) -> str:
        """Classify task type from query patterns"""
        scores = {task: 0 for task in self.task_patterns}
        
        for task, patterns in self.task_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    scores[task] += 1
        
        # Return task with highest score, default to vqa
        max_task = max(scores, key=scores.get)
        return max_task if scores[max_task] > 0 else 'vqa'
    
    def _extract_parameters(self, query: str, task_type: str) -> Dict:
        """Extract task-specific parameters"""
        parameters = {}
        
        if task_type == 'grounding':
            # Extract object to locate
            match = re.search(r'(where|find|locate)\s+(?:the\s+)?(\w+(?:\s+\w+)*)', query.lower())
            if match:
                parameters['target_object'] = match.group(2)
        
        elif task_type == 'change_detection':
            # Extract time references
            parameters['temporal'] = True
            if 'before' in query.lower() and 'after' in query.lower():
                parameters['comparison_type'] = 'before_after'
        
        elif task_type == 'sar_fusion':
            parameters['modalities'] = []
            if 'optical' in query.lower():
                parameters['modalities'].append('optical')
            if 'sar' in query.lower() or 'radar' in query.lower():
                parameters['modalities'].append('sar')
        
        return parameters
    
    def _extract_intent(self, query: str, task_type: str) -> str:
        """Extract high-level intent"""
        query_lower = query.lower()
        
        if 'count' in query_lower or 'how many' in query_lower:
            return 'counting'
        elif any(word in query_lower for word in ['where', 'locate', 'find']):
            return 'localization'
        elif any(word in query_lower for word in ['what', 'describe']):
            return 'description'
        elif any(word in query_lower for word in ['change', 'difference']):
            return 'comparison'
        else:
            return 'general_query'
    
    def _extract_entities(self, query: str) -> list:
        """Extract named entities and objects"""
        # Simple entity extraction (could be enhanced with NER models)
        entities = []
        
        # Common geographic/satellite entities
        entity_keywords = [
            'building', 'road', 'forest', 'water', 'field', 'urban', 'rural',
            'cloud', 'vegetation', 'infrastructure', 'river', 'lake', 'ocean'
        ]
        
        query_lower = query.lower()
        for entity in entity_keywords:
            if entity in query_lower:
                entities.append(entity)
        
        return entities
