"""
Native Ads Vector Store
FAISS-based vector store for RAG (Retrieval Augmented Generation)
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class NativeAdsVectorStore:
    """
    Vector store for native ads detection knowledge base.
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize vector store.
        
        Args:
            embedding_model: HuggingFace embedding model name
            persist_directory: Directory to save/load vector store
        """
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
        
        self.vectorstore = None
        
        # Load existing vector store if available
        if persist_directory and Path(persist_directory).exists():
            self.load()
    
    def create_from_dataset(self, dataset_path: str) -> None:
        """
        Create vector store from LLM dataset.
        
        Args:
            dataset_path: Path to JSON dataset file
        """
        logger.info(f"Creating vector store from: {dataset_path}")
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to documents
        documents = []
        for i, item in enumerate(data):
            # Extract content based on format
            reasoning = ""
            if 'input' in item:
                content = item['input']
                output_str = item.get('output', '')
                label = 'native ads' if 'native ads' in output_str.lower() else 'berita murni'
                
                # Extract reasoning from JSON output if available
                try:
                    if output_str.strip().startswith('{'):
                        output_json = json.loads(output_str)
                        reasoning = output_json.get('reasoning', '')
                except Exception:
                    pass
            elif 'context' in item:
                content = item['context']
                label = item.get('label', 'unknown')
                reasoning = item.get('reasoning', '')
            else:
                continue
            
            # Create document
            doc = Document(
                page_content=content[:1500],  # Phase 138: Deeper Indexing (500 -> 1500)
                metadata={
                    'id': i,
                    'label': label,
                    'reasoning': reasoning,
                    'source': 'training_dataset'
                }
            )
            documents.append(doc)
        
        logger.info(f"Creating vector store with {len(documents)} documents...")
        
        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(
            documents,
            self.embeddings
        )
        
        logger.info("Vector store created successfully")
        
        # Save if persist directory specified
        if self.persist_directory:
            self.save()
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to existing vector store.
        
        Args:
            documents: List of LangChain Documents
        """
        if self.vectorstore is None:
            logger.info("Creating new vector store")
            self.vectorstore = FAISS.from_documents(
                documents,
                self.embeddings
            )
        else:
            logger.info(f"Adding {len(documents)} documents to vector store")
            self.vectorstore.add_documents(documents)
        
        if self.persist_directory:
            self.save()
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_label: Optional[str] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results
            filter_label: Optional label filter ('native ads' or 'berita murni')
            
        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            logger.warning("Vector store not initialized")
            return []
        
        # Search
        if filter_label:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter={'label': filter_label}
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5
    ) -> List[tuple]:
        """
        Search with similarity scores.
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vectorstore is None:
            logger.warning("Vector store not initialized")
            return []
        
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def save(self) -> None:
        """Save vector store to disk."""
        if self.vectorstore is None:
            logger.warning("No vector store to save")
            return
        
        if self.persist_directory is None:
            logger.warning("No persist directory specified")
            return
        
        # Create directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        self.vectorstore.save_local(self.persist_directory)
        logger.info(f"Vector store saved to: {self.persist_directory}")
    
    def load(self) -> None:
        """Load vector store from disk."""
        if self.persist_directory is None:
            logger.warning("No persist directory specified")
            return
        
        if not Path(self.persist_directory).exists():
            logger.warning(f"Persist directory not found: {self.persist_directory}")
            return
        
        logger.info(f"Loading vector store from: {self.persist_directory}")
        self.vectorstore = FAISS.load_local(
            self.persist_directory,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully")
    
    def get_stats(self) -> dict:
        """Get vector store statistics."""
        if self.vectorstore is None:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'initialized',
            'embedding_model': self.embedding_model_name,
            'persist_directory': self.persist_directory,
            'index_size': self.vectorstore.index.ntotal if hasattr(self.vectorstore, 'index') else 'unknown'
        }
