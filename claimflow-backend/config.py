import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # App Settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'claimflow-secret-key')
    UPLOAD_FOLDER = 'uploads'
    DOCUMENT_FOLDER = os.getenv('CLAIMFLOW_DOCUMENT_FOLDER', 'DOCUMENT')
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff'}
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    # Using GPT-4o for best reasoning and Vision capabilities
    OPENAI_MODEL = "gpt-4o" 
    OPENAI_VISION_MODEL = os.getenv('OPENAI_VISION_MODEL', OPENAI_MODEL)
    OPENAI_TEMPERATURE = 0.1 # Low temperature for high accuracy/consistency

    # OCR Configuration
    # Update this path based on your OS (Windows/Mac/Linux)
    TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 
    MODEL_CACHE_DIR = os.path.abspath(os.getenv('CLAIMFLOW_MODEL_CACHE_DIR', 'storage/model_cache'))
    EASYOCR_MODULE_DIR = os.path.join(MODEL_CACHE_DIR, 'easyocr')
    EASYOCR_MODEL_DIR = os.path.join(EASYOCR_MODULE_DIR, 'model')
    EASYOCR_USER_NETWORK_DIR = os.path.join(EASYOCR_MODULE_DIR, 'user_network')
    HUGGINGFACE_CACHE_DIR = os.path.join(MODEL_CACHE_DIR, 'huggingface')
    PADDLE_CACHE_DIR = os.path.join(MODEL_CACHE_DIR, 'paddlex_v2')
    TROCR_MODEL = os.getenv('TROCR_MODEL', 'microsoft/trocr-base-handwritten')
    USE_TROCR_FULL_PAGE = os.getenv('USE_TROCR_FULL_PAGE', 'false').lower() == 'true'
    MAX_PAGES = int(os.getenv('CLAIMFLOW_MAX_PAGES', '3'))
    TEXTIN_APP_ID = os.getenv('TEXTIN_APP_ID')
    TEXTIN_SECRET_CODE = os.getenv('TEXTIN_SECRET_CODE')
    TEXTIN_DOCUMENT_URL = os.getenv(
        'TEXTIN_DOCUMENT_URL',
        'https://api.textin.com/ai/service/v2/recognize/document'
    )
    
    # Hardware Acceleration
    USE_CUDA = False  # Set to True if you have NVIDIA GPU and CUDA installed

    # Storage
    DB_FILE = 'storage/db.json'

    # Multi-Engine OCR Configuration
    MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
    MISTRAL_OCR_API_URL = os.getenv('MISTRAL_OCR_API_URL', 'https://api.mistral.ai/v1/ocr')
    USE_MISTRAL_PRIMARY = os.getenv('USE_MISTRAL_PRIMARY', 'true').lower() == 'true'
    OCR_CONFIDENCE_THRESHOLD = float(os.getenv('OCR_CONFIDENCE_THRESHOLD', '0.85'))
    ENABLE_VISION_GROUNDING = os.getenv('ENABLE_VISION_GROUNDING', 'true').lower() == 'true'
    
    # Optional external tool integrations
    NOTEBOOKLM_API_URL = os.getenv('NOTEBOOKLM_API_URL')
    NOTEBOOKLM_API_KEY = os.getenv('NOTEBOOKLM_API_KEY')
    OLMOCR_API_URL = os.getenv('OLMOCR_API_URL')
    PDFXCHANGE_BINARY_PATH = os.getenv('PDFXCHANGE_BINARY_PATH')

os.environ.setdefault('HF_HOME', Config.HUGGINGFACE_CACHE_DIR)
os.environ.setdefault('HF_HUB_CACHE', os.path.join(Config.HUGGINGFACE_CACHE_DIR, 'hub'))
os.environ.setdefault('TRANSFORMERS_CACHE', Config.HUGGINGFACE_CACHE_DIR)
os.environ.setdefault('EASYOCR_MODULE_PATH', Config.EASYOCR_MODULE_DIR)
os.environ.setdefault('PADDLE_PDX_CACHE_HOME', Config.PADDLE_CACHE_DIR)
