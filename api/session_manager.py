from typing import Dict, Optional
from datetime import datetime
import uuid


class SessionManager:
    """
    Manages user sessions and state across multiple queries
    """
    
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
    
    def create_session(self, session_id: str) -> dict:
        """Create a new session"""
        session = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'queries': [],
            'state': {},
            'metadata': {}
        }
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve session by ID"""
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, query_result: dict):
        """Update session with new query result"""
        if session_id in self.sessions:
            self.sessions[session_id]['queries'].append({
                'query_id': query_result['query_id'],
                'timestamp': datetime.now().isoformat(),
                'status': query_result['status']
            })
            self.sessions[session_id]['state'].update(query_result.get('state', {}))
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> list:
        """List all active sessions"""
        return list(self.sessions.keys())
