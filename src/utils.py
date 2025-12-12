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
    # Remove default handler
    logger.remove()
    
    # Add custom handlers
    logger.add(
        "logs/app.log",
        rotation="500 MB",
        retention="10 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    # Add console handler
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
            "host": "localhost",
            "port": 8001,
            "collection_name": "sentiment_embeddings"
        },
        "preprocessing": {
            "max_length": 512,
            "truncation": True,
            "padding": True
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge with default config
            for key in default_config:
                if key in user_config:
                    default_config[key].update(user_config[key])
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.warning(f"Error loading config: {e}. Using defaults.")
    else:
        logger.info("No config file found. Using default configuration.")
    
    return default_config

def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK data downloaded successfully")