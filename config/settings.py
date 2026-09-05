import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_PATH = os.getenv('MODELS_PATH', './models')
DATA_PATH = os.getenv('DATA_PATH', './data')

# API Configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))
API_KEY = os.getenv('API_KEY', 'default-api-key')

# Database
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/satquery_db')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Authentication
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
ALGORITHM = os.getenv('ALGORITHM', 'HS256')
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 30))

# Model Settings
VQA_MODEL = os.getenv('VQA_MODEL', 'blip-vqa-base')
GROUNDING_MODEL = os.getenv('GROUNDING_MODEL', 'groundingdino')
SAR_MODEL = os.getenv('SAR_MODEL', 'sentinel1-classifier')
CHANGE_DETECTION_MODEL = os.getenv('CHANGE_DETECTION_MODEL', 'siamese-net')

# Groq LLM Settings
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_MODEL_TEXT = os.getenv('GROQ_MODEL_TEXT', 'llama-3.3-70b-versatile')
GROQ_MODEL_VISION = os.getenv('GROQ_MODEL_VISION', 'llama-3.2-11b-vision-preview')

# Hugging Face Settings
HF_TOKEN = os.getenv('HF_TOKEN', '')
HF_VQA_MODEL = os.getenv('HF_VQA_MODEL', 'Salesforce/blip-vqa-base')
HF_GROUNDING_MODEL = os.getenv('HF_GROUNDING_MODEL', 'google/owlvit-base-patch32')
HF_FEATURE_MODEL = os.getenv('HF_FEATURE_MODEL', 'google/vit-base-patch16-224')

# Copernicus Data Space / Sentinel Hub (for real-time satellite imagery)
COPERNICUS_CLIENT_ID = os.getenv('COPERNICUS_CLIENT_ID', '')
COPERNICUS_CLIENT_SECRET = os.getenv('COPERNICUS_CLIENT_SECRET', '')
SENTINEL_HUB_BASE_URL = os.getenv('SENTINEL_HUB_BASE_URL',
    'https://sh.dataspace.copernicus.eu')

# Disaster Analysis Settings (Model 2)
DISASTER_TEMPORAL_LOOKBACK_DAYS = int(os.getenv('DISASTER_TEMPORAL_LOOKBACK_DAYS', 14))
FLOOD_NDWI_THRESHOLD = float(os.getenv('FLOOD_NDWI_THRESHOLD', 0.3))
SAR_FLOOD_THRESHOLD = float(os.getenv('SAR_FLOOD_THRESHOLD', -15))



# Processing Settings
MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', 1024))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 4))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.7))

# Supported image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.geotiff']

# Tool Registry
AVAILABLE_TOOLS = [
    'vqa_model',
    'grounding_model',
    'building_detector',
    'roboflow_building_detector',
    'spectral_index_model',
    'change_detection_model',
    'sar_fusion_model',
    'disaster_grounding_model',
]
