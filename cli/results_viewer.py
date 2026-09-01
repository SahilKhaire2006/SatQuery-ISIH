import json
from typing import Dict

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ResultsViewer:
    """
    Visual and textual results display
    """
    
    def __init__(self):
        pass
    
    def display_results(self, results: Dict):
        """
        Display query results in formatted way
        """
        print("\n" + "="*60)
        print("QUERY RESULTS")
        print("="*60)
        
        # Status
        status = results.get('status', 'unknown')
        print(f"\nStatus: {status.upper()}")
        
        if status == 'failed' or status == 'error':
            print(f"Error: {results.get('error', 'Unknown error')}")
            return
        
        # Main answer
        result_data = results.get('results', {})
        answer = result_data.get('answer', 'No answer available')
        print(f"\nAnswer:\n{answer}")
        
        # Confidence
        confidence = results.get('confidence', 0.0)
        print(f"\nOverall Confidence: {confidence:.2%}")
        
        # Individual tool confidences
        confidence_scores = result_data.get('confidence_scores', {})
        if confidence_scores:
            print("\nTool-specific Confidences:")
            for tool, score in confidence_scores.items():
                print(f"  - {tool}: {score:.2%}")
        
        # Details
        details = result_data.get('details', {})
        if details:
            print("\nDetailed Results:")
            for tool_id, tool_results in details.items():
                print(f"\n  {tool_id}:")
                if isinstance(tool_results, dict):
                    for key, value in tool_results.items():
                        if key not in ['visualization', 'change_map', 'fused_features']:
                            print(f"    {key}: {value}")
        
        # Audit log summary
        audit_log = results.get('audit_log', {})
        if audit_log.get('selected_tools'):
            tools = audit_log['selected_tools']
            print(f"\nTools Used: {len(tools)}")
            for tool in tools:
                print(f"  - {tool.get('tool_name', 'Unknown')}")
        
        print("\n" + "="*60 + "\n")
    
    def display_error(self, error_msg: str):
        """Display error message"""
        print("\n" + "="*60)
        print("ERROR")
        print("="*60)
        print(f"\n{error_msg}\n")
        print("="*60 + "\n")
    
    def export_results(self, results: Dict, output_file: str):
        """Export results to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults exported to: {output_file}")
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
