import numpy as np
from typing import Dict, Any, Optional
from models.llm.groq_engine import GroqLLMEngine
from visualization.evidence_compiler import EvidenceCompiler
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ResultAggregator:
    """
    Synthesizes model outputs into evidence-grounded natural language responses & visual evidence (USP-2)
    """

    def __init__(self, llm_engine: Optional[GroqLLMEngine] = None):
        self.llm_engine = llm_engine or GroqLLMEngine()
        self.evidence_compiler = EvidenceCompiler()

    async def aggregate(
        self,
        query: str,
        interpretation: Dict[str, Any],
        tool_results: Dict[str, Any],
        image: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Aggregate multi-model tool results into final synthesized answer, textual evidence, and visual evidence overlays.
        """
        logger.info(f"Aggregating results for query: '{query}'")

        # Extract tool details if wrapped inside execution_engine aggregated structure
        details = tool_results.get('details', tool_results)

        # Call Groq LLM aggregator (or fallback)
        llm_aggregation = await self.llm_engine.aggregate_results(
            query=query,
            interpretation=interpretation,
            tool_results=details
        )
        
        # Extract summary text from LLM response
        if isinstance(llm_aggregation, dict):
            summary_text = llm_aggregation.get('summary', 'Analysis complete.')
        else:
            summary_text = str(llm_aggregation)

        # Build evidence inventory for USP-2 audit trail
        evidence = []
        vqa_answer = None
        grounding_count = 0
        
        for tool_id, res in details.items():
            if not isinstance(res, dict):
                continue
            
            out = res.get('output', res)
            conf = res.get('confidence', out.get('confidence', 0.0))
            
            # Track VQA answer
            if tool_id == 'vqa_model':
                vqa_answer = out.get('answer', '')
            
            # Track grounding detections
            if tool_id == 'grounding_model' and 'detections' in out:
                grounding_count = len(out['detections'])
            
            ev_item = {
                'source_tool': tool_id,
                'confidence': float(conf) if isinstance(conf, (int, float)) else 0.0,
                'execution_time': res.get('execution_time', 0.0)
            }
            if 'detections' in out:
                ev_item['type'] = 'bounding_boxes'
                ev_item['details'] = out['detections']
            elif 'change_percentage' in out:
                ev_item['type'] = 'change_map'
                ev_item['details'] = {'change_percentage': out['change_percentage']}
            elif 'fusion_info' in out:
                ev_item['type'] = 'multi_modal_fusion'
                ev_item['details'] = out['fusion_info']
            else:
                ev_item['type'] = 'vqa_answer'
                ev_item['details'] = {'answer': out.get('answer', str(out))}

            evidence.append(ev_item)
        
        # Check for VQA/grounding disagreement
        confidence_adjustment = 1.0
        if vqa_answer and grounding_count > 0:
            # If VQA says "no" but grounding found objects, flag disagreement
            if any(word in vqa_answer.lower() for word in ['no', 'not', 'none', 'zero', 'absent']):
                logger.warning(f"VQA/grounding disagreement detected: VQA='{vqa_answer}' but found {grounding_count} detections")
                confidence_adjustment = 0.5
        elif vqa_answer and grounding_count == 0:
            # If VQA says "yes" but grounding found nothing, also flag
            if not any(word in vqa_answer.lower() for word in ['no', 'not', 'none', 'zero', 'absent']):
                logger.warning(f"VQA/grounding disagreement: VQA found objects but grounding detected 0")
                confidence_adjustment = 0.6

        # Generate visual evidence overlays if image is supplied
        visual_package = {}
        if image is not None:
            visual_package = self.evidence_compiler.compile_evidence(image, details, interpretation)

        # Apply confidence adjustment if disagreement detected
        final_confidence = interpretation.get('confidence', 0.9) * confidence_adjustment

        aggregated_response = {
            'summary': summary_text,
            'evidence': evidence,
            'visual_evidence': visual_package.get('visual_outputs', {}),
            'evidence_records': visual_package.get('evidence_records', []),
            'interpretation_summary': {
                'task_type': interpretation.get('task_type'),
                'intent': interpretation.get('intent'),
                'confidence': final_confidence
            }
        }

        return aggregated_response
