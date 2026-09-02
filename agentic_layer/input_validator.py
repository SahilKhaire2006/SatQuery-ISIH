from typing import Dict, Optional
import io
from PIL import Image
import numpy as np

from config.settings import MAX_IMAGE_SIZE, SUPPORTED_FORMATS
from utils.logger import setup_logger

logger = setup_logger(__name__)


class InputValidator:
    """
    Validates input format, quality, geo-metadata, and parsing
    """
    
    def __init__(self):
        self.max_size = MAX_IMAGE_SIZE
        self.supported_formats = SUPPORTED_FORMATS
    
    async def validate(
        self,
        query: str,
        image_data: bytes,
        image_filename: str,
        geo_metadata: Optional[str] = None
    ) -> Dict:
        """
        Validate all inputs
        """
        errors = []
        warnings = []
        
        # Validate query
        if not query or len(query.strip()) == 0:
            errors.append("Query cannot be empty")
        elif len(query) > 1000:
            errors.append("Query too long (max 1000 characters)")
        
        # Validate image
        try:
            if image_data and len(image_data) > 0:
                image = Image.open(io.BytesIO(image_data))
                
                # Check format
                file_ext = '.' + image_filename.split('.')[-1].lower() if image_filename else '.png'
                if file_ext not in self.supported_formats:
                    warnings.append(f"Unusual image format: {file_ext}")
                
                # Check size
                width, height = image.size
                if width > self.max_size or height > self.max_size:
                    warnings.append(f"Image will be resized from {width}x{height} to max {self.max_size}px")
                    image = self._resize_image(image)
                
                # Convert to array
                processed_image = np.array(image)
            else:
                # No image uploaded - will be fetched via location in query / geocoder
                processed_image = None
                warnings.append("No uploaded image provided. Location/geocoding tile fetch active.")
            
        except Exception as e:
            processed_image = None
            warnings.append(f"Uploaded image invalid ({e}). Location tile fetch active.")
        
        # Validate geo-metadata (if provided)
        geo_valid = True
        if geo_metadata:
            try:
                # Parse and validate geo metadata
                # This would include coordinate validation, projection checks, etc.
                pass
            except Exception as e:
                warnings.append(f"Invalid geo-metadata: {str(e)}")
                geo_valid = False
        
        # Determine task type from query
        task_type = self._classify_task(query)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'processed_image': processed_image,
            'image_filename': image_filename,
            'image_shape': processed_image.shape if processed_image is not None else None,
            'geo_metadata_valid': geo_valid,
            'task_type': task_type
        }
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image maintaining aspect ratio"""
        width, height = image.size
        if width > height:
            new_width = self.max_size
            new_height = int(height * (self.max_size / width))
        else:
            new_height = self.max_size
            new_width = int(width * (self.max_size / height))
        
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    def _classify_task(self, query: str) -> str:
        """Classify task type from query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['change', 'difference', 'compare']):
            return 'change_detection'
        elif any(word in query_lower for word in ['locate', 'find', 'where', 'detect', 'playground', 'building', 'structure', 'grounding', 'search', 'reservoir', 'water body']):
            return 'grounding'
        elif any(word in query_lower for word in ['sar', 'radar', 'fusion']):
            return 'sar_fusion'
        else:
            return 'vqa'
