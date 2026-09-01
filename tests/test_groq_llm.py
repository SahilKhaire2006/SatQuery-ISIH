import sys
import os
import asyncio
import pytest

# Ensure root dir is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.llm.groq_engine import GroqLLMEngine
from agentic_layer.query_interpreter import QueryInterpreter
from agentic_layer.tool_selector import ToolSelector
from agentic_layer.result_aggregator import ResultAggregator
from agentic_layer.orchestrator import AgenticOrchestrator


def test_groq_llm_engine_initialization():
    async def _run():
        engine = GroqLLMEngine()
        assert engine is not None
        res = await engine.interpret_query("Locate buildings near river")
        assert 'task_type' in res
        assert res['task_type'] in ['grounding', 'vqa', 'change_detection', 'sar_fusion']

    asyncio.run(_run())


def test_query_interpreter():
    async def _run():
        interpreter = QueryInterpreter()
        grounding_res = await interpreter.interpret("Where is the lake?")
        assert grounding_res['task_type'] in ['grounding', 'vqa']

        sar_res = await interpreter.interpret("Process Sentinel-1 SAR radar image")
        assert sar_res['task_type'] == 'sar_fusion' or 'sar' in sar_res.get('modalities', [])

    asyncio.run(_run())


def test_tool_selector_usp1_sar_routing():
    async def _run():
        selector = ToolSelector()
        interpretation = {
            'task_type': 'sar_fusion',
            'original_query': 'Analyze SAR radar data',
            'intent': 'fusion',
            'modalities': ['optical', 'sar'],
            'parameters': {'modalities': ['optical', 'sar']}
        }
        tools = await selector.select_tools(interpretation, task_type='sar_fusion')
        assert len(tools) > 0
        assert tools[0]['tool_id'] == 'sar_fusion_model'

    asyncio.run(_run())


def test_result_aggregator():
    async def _run():
        aggregator = ResultAggregator()
        query = "How many ships are in the harbor?"
        interpretation = {
            'task_type': 'vqa',
            'intent': 'counting',
            'confidence': 0.95
        }
        tool_results = {
            'vqa_model': {
                'output': {'answer': '5 ships detected', 'confidence': 0.92},
                'confidence': 0.92,
                'execution_time': 0.15
            }
        }
        result = await aggregator.aggregate(query, interpretation, tool_results)
        assert 'summary' in result
        assert len(result['evidence']) == 1
        assert result['evidence'][0]['source_tool'] == 'vqa_model'

    asyncio.run(_run())


def test_orchestrator_end_to_end():
    async def _run():
        orchestrator = AgenticOrchestrator()
        import cv2
        import numpy as np
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', dummy_img)
        image_bytes = buffer.tobytes()

        response = await orchestrator.process_request(
            session_id="test_session_123",
            query="Locate the building in this satellite image",
            image_data=image_bytes,
            image_filename="test_image.png"
        )

        assert response['status'] == 'success'
        assert 'explanation' in response
        assert 'results' in response
        assert 'audit_log' in response
        assert response['audit_log']['validation']['valid'] is True

    asyncio.run(_run())


if __name__ == "__main__":
    test_groq_llm_engine_initialization()
    test_query_interpreter()
    test_tool_selector_usp1_sar_routing()
    test_result_aggregator()
    test_orchestrator_end_to_end()
    print("All Phase 2 tests passed successfully!")
