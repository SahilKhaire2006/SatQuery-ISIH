import os
from pathlib import Path
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageUploader:
    """
    Handles image file selection and upload
    Supports: GeoTIFF, TIFF, PNG, JPEG
    """
    
    def __init__(self):
        self.supported_formats = ['.tif', '.tiff', '.geotiff', '.png', '.jpg', '.jpeg']
    
    def select_image(self, file_path: str) -> Optional[dict]:
        """
        Select and validate image file
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        if not path.is_file():
            logger.error(f"Not a file: {file_path}")
            return None
        
        # Check extension
        if path.suffix.lower() not in self.supported_formats:
            logger.error(f"Unsupported format: {path.suffix}")
            return None
        
        # Read file
        try:
            with open(path, 'rb') as f:
                image_data = f.read()
            
            return {
                'filename': path.name,
                'path': str(path),
                'size': len(image_data),
                'data': image_data,
                'format': path.suffix
            }
        
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            return None
    
    def list_images_in_directory(self, directory: str) -> list:
        """
        List all supported images in directory
        """
        dir_path = Path(directory)
        
        if not dir_path.exists() or not dir_path.is_dir():
            return []
        
        images = []
        for ext in self.supported_formats:
            images.extend(dir_path.glob(f"*{ext}"))
        
        return [str(img) for img in images]
