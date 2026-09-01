from typing import Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)


class QueryConsole:
    """
    Console interface for query input
    """
    
    def __init__(self):
        self.history = []
    
    def get_query(self, prompt: str = "Enter your query: ") -> str:
        """
        Get query from user input
        """
        try:
            query = input(prompt).strip()
            if query:
                self.history.append(query)
            return query
        except (KeyboardInterrupt, EOFError):
            return ""
    
    def get_multiline_query(self) -> str:
        """
        Get multi-line query (end with empty line)
        """
        print("Enter your query (press Enter twice to finish):")
        lines = []
        
        try:
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            
            query = " ".join(lines).strip()
            if query:
                self.history.append(query)
            return query
        
        except (KeyboardInterrupt, EOFError):
            return ""
    
    def show_history(self):
        """Display query history"""
        if not self.history:
            print("No query history")
            return
        
        print("\n=== Query History ===")
        for idx, query in enumerate(self.history, 1):
            print(f"{idx}. {query}")
        print()
    
    def get_previous_query(self, index: int = -1) -> Optional[str]:
        """Get query from history"""
        try:
            return self.history[index]
        except IndexError:
            return None
