import os
import json
import hashlib
from typing import Dict, Any, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from config.settings import REDIS_URL
from utils.logger import setup_logger

logger = setup_logger(__name__)


class QueryCache:
    """
    Production Query Result Cache supporting Redis & In-Memory storage.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QueryCache, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._memory_cache: Dict[str, Any] = {}
        self.redis_client = None

        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(REDIS_URL, socket_timeout=2)
                self.redis_client.ping()
                logger.info("Connected to Redis cache server successfully.")
            except Exception as e:
                logger.info(f"Redis connection unavailable ({e}). Using in-memory query cache.")

        self._initialized = True

    def _generate_key(self, query: str, image_bytes: bytes) -> str:
        """Generate MD5 hash key from query and image bytes"""
        hasher = hashlib.md5()
        hasher.update(query.lower().strip().encode('utf-8'))
        if image_bytes:
            hasher.update(image_bytes[:1024])  # Hash first 1KB of image for speed
        return f"satquery:cache:{hasher.hexdigest()}"

    def get(self, query: str, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Retrieve cached query response"""
        key = self._generate_key(query, image_bytes)

        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    logger.info("Query response loaded from Redis cache.")
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")

        if key in self._memory_cache:
            logger.info("Query response loaded from in-memory cache.")
            return self._memory_cache[key]

        return None

    def set(self, query: str, image_bytes: bytes, result: Dict[str, Any], ttl_seconds: int = 3600):
        """Store query response in cache"""
        key = self._generate_key(query, image_bytes)

        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl_seconds, json.dumps(result))
            except Exception as e:
                logger.warning(f"Redis set error: {e}")

        self._memory_cache[key] = result
