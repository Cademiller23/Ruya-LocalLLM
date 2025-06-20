import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
COHERE_API_KEY = os.getenv('COHERE_API_KEY', os.getenv('cohere_API_KEY'))  # Support both cases
MEM0AI_API_KEY = os.getenv('MEM0AI_API_KEY', os.getenv('mem0AI_API_KEY'))  # Support both cases

# Qdrant Configuration
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', None)

# Server Configuration
PORT = int(os.getenv('PORT', 5002))
HOST = os.getenv('HOST', '0.0.0.0')

# Database Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///memories.db')

# Memory Configuration
MEMORY_THRESHOLD = float(os.getenv('MEMORY_THRESHOLD', 0.25))
MAX_MEMORIES = int(os.getenv('MAX_MEMORIES', 10))
RERANK_TOP_N = int(os.getenv('RERANK_TOP_N', 5))

# Memory Type Settings
SHORT_TERM_DAYS = int(os.getenv('SHORT_TERM_DAYS', 7))
ACCESS_COUNT_THRESHOLD = int(os.getenv('ACCESS_COUNT_THRESHOLD', 3))


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
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-english-v2.0")



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



# Narrative Summary Configuration
NARRATIVE_SUMMARY_CONFIG = {
    "min_memories": 5,
    "max_memories": 20,
    "update_frequency": 10,  # Update after every N memories
    "max_tokens": 150
}

# User Trait Extraction Configuration
TRAIT_EXTRACTION_CONFIG = {
    "extract_name": True,
    "extract_location": True,
    "extract_role": True,
    "extract_timezone": True,
    "analyze_sentiment": True,
    "sentiment_keywords": {
        "positive": ['happy', 'great', 'love', 'excellent', 'wonderful', 'amazing', 'fantastic'],
        "negative": ['sad', 'angry', 'hate', 'terrible', 'awful', 'horrible', 'frustrated']
    }
}

# Thread Configuration
THREAD_CONFIG = {
    "auto_save": True,
    "save_interval": 5,  # Save after every N messages
    "max_thread_age_days": 365,  # Keep threads for 1 year
    "cleanup_enabled": False
}

# Reranking Configuration
RERANK_CONFIG = {
    "enabled": True,
    "model": "rerank-english-v2.0",
    "top_n": 5,
    "min_relevance_score": 0.3
}

# Long-term/Short-term Memory Configuration
MEMORY_TYPE_CONFIG = {
    "important_keywords": [
        'name is', 'i am', 'my goal', 'important', 'remember', 
        'always', 'never forget', 'key point', 'critical'
    ],
    "promotion_rules": {
        "access_count": 3,
        "age_days": 7,
        "minimum_accesses_for_promotion": 1
    }
}