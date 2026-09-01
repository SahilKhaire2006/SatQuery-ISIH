#!/usr/bin/env python3
"""
SatQuery - Satellite Image Query System
Main API server entry point
"""

import uvicorn
from config.settings import API_HOST, API_PORT
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """
    Start the API server
    """
    logger.info("Starting SatQuery API Server")
    logger.info(f"Server will run on http://{API_HOST}:{API_PORT}")
    logger.info("API documentation available at http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(
        "api.gateway:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
