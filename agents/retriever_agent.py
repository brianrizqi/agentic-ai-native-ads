"""
Retriever Agent Module
Handles document retrieval using vector similarity search.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """
    Retriever Agent yang mengambil dokumen relevan dari knowledge base.
    """
    
    def __init__(self, vector_db_path: Optional[str] = None, 
                 embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 top_k: int = 5):
        """
        Initialize Retriever Agent.
        """
        self.vector_db_path = vector_db_path
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self.documents = []  # In-memory document store
        self.embeddings = []  # In-memory embeddings
        self.embedding_model = self._load_embedding_model()
        
        # Load initial data if path provided
        if vector_db_path:
            self.load_knowledge_base(vector_db_path)
            
        logger.info(f"Retriever Agent initialized with {embedding_model}")
    
    def _load_embedding_model(self):
        """Load embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully")
            return model
        except ImportError:
            logger.warning("sentence-transformers not installed (pip install sentence-transformers)")
            return None
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            return None
    
    def retrieve(self, preprocessed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        logger.info("Retrieving relevant documents")
        
        query_text = preprocessed_data.get('cleaned_text', '')
        
        if not self.documents:
            logger.warning("No documents in knowledge base")
            return []
        
        if self.embedding_model:
            # Use actual embeddings
            query_embedding = self.embedding_model.encode([query_text])[0]
            
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(self.embeddings):
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-k documents
            results = []
            for idx, score in similarities[:self.top_k]:
                doc = self.documents[idx]
                results.append({
                    'id': doc.get('id', f'doc_{idx}'),
                    'content': doc.get('content', str(doc)),
                    'label': doc.get('label', 'unknown'),
                    'relevance_score': float(score),
                    'metadata': {
                        'index': idx,
                        'retrieval_method': 'vector_similarity'
                    }
                })
        else:
            # Fallback
            results = self._keyword_based_retrieval(query_text)
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to knowledge base."""
        logger.info(f"Adding {len(documents)} documents to knowledge base")
        
        self.documents.extend(documents)
        
        if self.embedding_model:
            # Extract content for embedding
            texts = [doc.get('content', str(doc)) for doc in documents]
            new_embeddings = self.embedding_model.encode(texts)
            self.embeddings.extend(new_embeddings)
        
        logger.info(f"Total documents in knowledge base: {len(self.documents)}")
        
    def load_knowledge_base(self, file_path: str):
        """Load knowledge base from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.add_documents(data)
                else:
                    logger.warning("Invalid knowledge base format (expected list)")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
    
    def _keyword_based_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Fallback keyword retrieval."""
        query_words = set(query.lower().split())
        scores = []
        
        for i, doc in enumerate(self.documents):
            content = doc.get('content', str(doc))
            doc_words = set(content.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / len(query_words) if query_words else 0
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:self.top_k]:
            if score > 0:
                doc = self.documents[idx]
                results.append({
                    'id': doc.get('id', f'doc_{idx}'),
                    'content': doc.get('content', ''),
                    'relevance_score': float(score)
                })
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            'total_documents': len(self.documents),
            'embedding_model': self.embedding_model_name,
            'top_k': self.top_k,
            'has_embeddings': len(self.embeddings) > 0
        }
