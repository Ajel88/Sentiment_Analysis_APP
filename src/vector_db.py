import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import uuid
import numpy as np
import time
import logging
import os  # ADD THIS IMPORT
from .embedding_generator import EmbeddingResult

logger = logging.getLogger(__name__)

class VectorDatabase:
    """ChromaDB vector database manager with connection retry"""
    
    def __init__(self, host: str = None, port: int = None, 
                 collection_name: str = "sentiment_embeddings"):
        # GET FROM ENVIRONMENT VARIABLES OR USE PROVIDED VALUES
        self.host = host or os.getenv("CHROMA_HOST", "chroma-db")  # FIXED
        self.port = port or int(os.getenv("CHROMA_PORT", "8000"))  # FIXED: 8000 not 8001
        self.collection_name = collection_name
        
        logger.info(f"Attempting to connect to ChromaDB at {self.host}:{self.port}")
        
        # Connect with retry
        self.client = self._connect_with_retry()
        self.collection = self.get_or_create_collection()
        
        logger.info(f"Successfully connected to ChromaDB at {self.host}:{self.port}")
    
    def _connect_with_retry(self, max_retries: int = 10, delay: int = 5):  # Increased retries
        """Connect to ChromaDB with retry logic"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Connecting to {self.host}:{self.port}")
                client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    settings=Settings(
                        allow_reset=True,
                        anonymized_telemetry=False
                    )
                )
                # Test connection
                heartbeat = client.heartbeat()
                logger.info(f"Connection successful! Heartbeat: {heartbeat}")
                return client
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to connect to ChromaDB after {max_retries} attempts")
                    logger.info("Trying alternative connection methods...")
                    
                    # Try alternative hostnames
                    alternative_hosts = ["chroma-db", "localhost", "host.docker.internal"]
                    for alt_host in alternative_hosts:
                        try:
                            logger.info(f"Trying alternative host: {alt_host}")
                            client = chromadb.HttpClient(
                                host=alt_host,
                                port=self.port,
                                settings=Settings(allow_reset=True)
                            )
                            client.heartbeat()
                            self.host = alt_host  # Update host
                            logger.info(f"Connected using alternative host: {alt_host}")
                            return client
                        except Exception as alt_e:
                            logger.debug(f"Alternative {alt_host} also failed: {alt_e}")
                    
                    raise ConnectionError(f"Could not connect to ChromaDB at {self.host}:{self.port} or any alternative")
    
    def get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            collection = self.client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
            return collection
        except Exception as e:
            logger.info(f"Creating new collection: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Sentiment analysis embeddings",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "app": "Sentiment Analysis System"
                }
            )
    
    def store_embeddings(self, embeddings_results: List[EmbeddingResult]):
        """Store embeddings in ChromaDB"""
        try:
            ids = [str(uuid.uuid4()) for _ in range(len(embeddings_results))]
            embeddings = [result.embedding.tolist() for result in embeddings_results]
            metadatas = [result.metadata for result in embeddings_results]
            documents = [result.text for result in embeddings_results]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                ids=ids
            )
            
            logger.info(f"Stored {len(embeddings_results)} embeddings in ChromaDB")
            return ids
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5, 
                      where_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings in ChromaDB"""
        try:
            # Prepare query
            query_embedding_list = query_embedding.tolist()
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            
            # Check if we have results
            if (results['documents'] and 
                len(results['documents']) > 0 and 
                len(results['documents'][0]) > 0):
                
                for i in range(len(results['documents'][0])):
                    doc = results['documents'][0][i]
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    
                    formatted_results.append({
                        'document': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance  # Convert distance to similarity
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {e}")
            return []
    
    def get_all_embeddings(self) -> tuple:
        """Get all embeddings from collection"""
        try:
            results = self.collection.get(include=['embeddings', 'metadatas', 'documents'])
            return results['embeddings'], results['metadatas'], results['documents']
        except Exception as e:
            logger.error(f"Error getting all embeddings: {e}")
            return [], [], []
    
    def reset_collection(self):
        """Reset the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.get_or_create_collection()
            logger.info(f"Collection {self.collection_name} reset successfully")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "host": self.host,
                "port": self.port,
                "status": "Connected"
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e), "status": "Disconnected"}