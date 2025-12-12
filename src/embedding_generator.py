import torch
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    text: str
    embedding: np.ndarray
    metadata: dict

class EmbeddingGenerator:
    """Generate embeddings for text using transformer models"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading embedding model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name
        logger.info(f"Embedding model loaded successfully")
    
    def generate_embedding(self, text: str, metadata: dict = None) -> EmbeddingResult:
        """Generate embedding for a single text"""
        if metadata is None:
            metadata = {}
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Add text to metadata
            metadata['text'] = text[:100] + "..." if len(text) > 100 else text
            metadata['model'] = self.model_name
            metadata['embedding_dim'] = embedding.shape[0]
            
            return EmbeddingResult(
                text=text,
                embedding=embedding,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return empty embedding on error
            empty_embedding = np.zeros(384)  # Default dimension for MiniLM
            return EmbeddingResult(
                text=text,
                embedding=empty_embedding,
                metadata=metadata
            )
    
    def generate_batch_embeddings(self, texts: List[str], metadatas: List[dict] = None) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts"""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        try:
            # Generate embeddings in batch
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            
            results = []
            for text, embedding, metadata in zip(texts, embeddings, metadatas):
                metadata['text'] = text[:100] + "..." if len(text) > 100 else text
                metadata['model'] = self.model_name
                metadata['embedding_dim'] = embedding.shape[0]
                
                results.append(EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    metadata=metadata
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            # Return empty embeddings on error
            return [
                EmbeddingResult(
                    text=text,
                    embedding=np.zeros(384),
                    metadata=metadata
                )
                for text, metadata in zip(texts, metadatas)
            ]