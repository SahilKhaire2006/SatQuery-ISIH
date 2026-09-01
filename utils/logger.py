import logging
import sys
from datetime import datetime


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Setup logger with consistent formatting
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Only add handler if not already added
    if not logger.handlers:
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
    
    return logger
