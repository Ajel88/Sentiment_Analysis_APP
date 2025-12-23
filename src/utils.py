"""
Utility functions for sentiment analysis system
"""
import json
import os
from typing import Dict, Any
from loguru import logger
import nltk

def setup_logging():
    """Configure logging with loguru"""
    # Setup NLTK quietly
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        logger.info("NLTK data ready")
    except Exception as e:
        logger.warning(f"NLTK note: {e}")
    
    # Remove default handler
    logger.remove()
    
    # Add console handler only
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )
    
    return logger

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    default_config = {
        "model": {
            "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
            "embedding": "all-MiniLM-L6-v2"
        },
        "chroma": {
            "host": "chroma-db",  # Docker service name
            "port": 8000,         # Docker container port
            "collection_name": "sentiment_embeddings"
        }
    }
    
    # ENVIRONMENT VARIABLES TAKE PRIORITY
    chroma_host = os.getenv('CHROMA_HOST')
    chroma_port = os.getenv('CHROMA_PORT')
    
    if chroma_host:
        default_config['chroma']['host'] = chroma_host
        logger.info(f"Using CHROMA_HOST from environment: {chroma_host}")
    
    if chroma_port:
        default_config['chroma']['port'] = int(chroma_port)
        logger.info(f"Using CHROMA_PORT from environment: {chroma_port}")
    
    return default_config

def download_nltk_data():
    """Download required NLTK data"""
    setup_logging()  # This will download NLTK