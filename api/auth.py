import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
from pydantic import BaseModel

try:
    from jose import JWTError, jwt
    JOSE_AVAILABLE = True
except ImportError:
    JOSE_AVAILABLE = False

from config.settings import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class User(BaseModel):
    username: str
    email: Optional[str] = None
    disabled: Optional[bool] = None


# Mock user store for production API testing
MOCK_USERS_DB = {
    "admin": {
        "username": "admin",
        "email": "admin@satquery.ai",
        "hashed_password": "secret_admin_password",
        "disabled": False
    }
}


def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Generate JWT access token.
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": int(expire.timestamp())})
    
    if JOSE_AVAILABLE:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    else:
        import base64
        import json
        payload_str = json.dumps(to_encode)
        b64 = base64.b64encode(payload_str.encode()).decode()
        return f"mock_jwt_{b64}"


def verify_token(token: str) -> Optional[Dict]:
    """
    Verify and decode JWT token.
    """
    if JOSE_AVAILABLE:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            logger.warning(f"JWT Verification failed: {e}")
            return None
    else:
        if token.startswith("mock_jwt_"):
            try:
                import base64
                import json
                b64 = token.replace("mock_jwt_", "")
                payload_str = base64.b64decode(b64.encode()).decode()
                return json.loads(payload_str)
            except Exception:
                return None
        return None
