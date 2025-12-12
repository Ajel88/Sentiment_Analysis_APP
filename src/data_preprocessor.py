import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Tuple, Dict
import pandas as pd
from transformers import AutoTokenizer

class TextPreprocessor:
    """Text preprocessing pipeline for sentiment analysis"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing special characters, URLs, etc."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers (keep basic punctuation)
        text = re.sub(r'[^a-zA-Z\s\.!?,]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text"""
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def tokenize(self, text: str, max_length: int = 512) -> Dict:
        """Tokenize text for transformer model"""
        return self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
    
    def preprocess_pipeline(self, text: str) -> Tuple[str, Dict]:
        """Complete preprocessing pipeline"""
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize (don't remove stopwords for transformer models)
        tokens = self.tokenize(cleaned)
        
        return cleaned, tokens
    
    def batch_preprocess(self, texts: List[str]) -> Tuple[List[str], List[Dict]]:
        """Preprocess multiple texts"""
        cleaned_texts = []
        tokenized_texts = []
        
        for text in texts:
            cleaned, tokens = self.preprocess_pipeline(text)
            cleaned_texts.append(cleaned)
            tokenized_texts.append(tokens)
        
        return cleaned_texts, tokenized_texts