import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
COHERE_API_KEY = os.getenv('COHERE_API_KEY', os.getenv('cohere_API_KEY'))  # Support both cases
MEM0AI_API_KEY = os.getenv('MEM0AI_API_KEY', os.getenv('mem0AI_API_KEY'))  # Support both cases


# Server Configuration
PORT = int(os.getenv('PORT', 5002))
HOST = os.getenv('HOST', '0.0.0.0')

# Database Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///memories.db')

# Memory Configuration
MEMORY_THRESHOLD = float(os.getenv('MEMORY_THRESHOLD', 0.25))
MAX_MEMORIES = int(os.getenv('MAX_MEMORIES', 10))

# Validate required environment variables
if not COHERE_API_KEY:
    raise ValueError("Cohere API key is not set. Please set the COHERE_API_KEY or cohere_API_KEY environment variable.")

# Optional configurations
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Base paths
BASE_DIR = Path(__file__).parent 
MEMORY_DB_PATH = BASE_DIR / "memory_db"
MEMORY_DB_PATH.mkdir(exist_ok=True)

# API Configuration
PYTHON_MEMORY_HOST = os.getenv("PYTHON_MEMORY_HOST", "127.0.0.1")
PYTHON_MEMORY_PORT = int(os.getenv("PYTHON_MEMORY_PORT", 5002))

# Validate Mem0AI API key if needed
if not MEM0AI_API_KEY:
    print("Warning: Mem0AI API key is not set. Some features may not work.")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embed-english-v3.0")

# Memory configuration for mem0AI
MEM0_CONFIG = {
    "vector_store": {
        "provider": 'chroma',
        "config": {
            'collection_name': "ruya_ai_memories",
            'path': str(MEMORY_DB_PATH / "chroma"),
            "embedding_function": None
        }
    },
    "llm": {
        "provider": 'ollama',
        "config": {
            "model": 'llama3',
            "base_url": 'http://localhost:11434',
            "temperature": 0.1,
        }
    },
    "embedder": {
        "provider": 'cohere',
        "config": {
            "api_key": COHERE_API_KEY,
            "model": EMBEDDING_MODEL
        }
    },
    "version": 'v1.0'
}

# Hybrid Search config
HYBRID_SEARCH_CONFIG = {
    "alpha": 0.5, # Balance between lexical (0) and semantic (1) search
    "max_results": 20,
    "min_score_threshold": 0.3,
    "lexical_weight": 0.4,
    "semantic_weight": 0.6,
}

# Text Processing Configuration
TEXT_PROCESSING = {
    "min_token_length": 4,
    "remove_stopwords": True,
    "use_stemming": True,
    "language": "english"
}