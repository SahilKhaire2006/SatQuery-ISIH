import numpy as np
from typing import Any

def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively converts non-JSON-serializable objects (numpy arrays, numpy scalars,
    bytes, sets, custom objects) into native Python types (int, float, bool, list, dict, str).
    """
    if obj is None:
        return None
    elif isinstance(obj, (bool, str, int, float)):
        return obj
    elif isinstance(obj, np.ndarray):
        # Large image arrays (e.g. masks, images) should not be embedded as raw numeric lists in JSON responses
        if obj.size <= 100:
            return obj.tolist()
        else:
            return f"<ndarray shape={list(obj.shape)} dtype={str(obj.dtype)}>"
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except Exception:
            return f"<bytes len={len(obj)}>"
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return sanitize_for_json(obj.__dict__)
    else:
        try:
            return str(obj)
        except Exception:
            return "<unserializable>"
