"""
Retrieval Tools for LangChain
Provides tools for document retrieval using vector similarity search
"""

from typing import Optional, Type, List, Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
import logging

logger = logging.getLogger(__name__)


class VectorSearchInput(BaseModel):
    """Input schema for VectorSearchTool."""
    query: str = Field(description="Query text to search for similar documents")
    top_k: int = Field(default=5, description="Number of top results to return")


class VectorSearchTool(BaseTool):
    """Tool for searching documents using vector similarity."""
    
    name: str = "vector_search"
    description: str = """
    Searches for similar documents using vector embeddings.
    Input is a query string and optional top_k parameter.
    Returns a list of relevant documents with similarity scores.
    """
    args_schema: Type[BaseModel] = VectorSearchInput
    vectorstore: Any = None  # Will be injected
    
    def __init__(self, vectorstore=None, **kwargs):
        super().__init__(**kwargs)
        self.vectorstore = vectorstore
    
    def _run(
        self,
        query: str,
        top_k: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            if self.vectorstore is None:
                logger.warning("No vectorstore configured")
                return []
            
            # Search using vectorstore
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=top_k
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score)
                })
            
            logger.info(f"Retrieved {len(formatted_results)} documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []


class KeywordSearchInput(BaseModel):
    """Input schema for KeywordSearchTool."""
    query: str = Field(description="Query text for keyword search")
    documents: List[Dict[str, Any]] = Field(description="List of documents to search")
    top_k: int = Field(default=5, description="Number of results to return")


class KeywordSearchTool(BaseTool):
    """Tool for keyword-based document search (fallback)."""
    
    name: str = "keyword_search"
    description: str = """
    Searches documents using keyword matching (fallback when vector search unavailable).
    Input is query text and a list of documents.
    Returns top matching documents based on keyword overlap.
    """
    args_schema: Type[BaseModel] = KeywordSearchInput
    
    def _run(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[Dict[str, Any]]:
        """Keyword-based search."""
        try:
            query_words = set(query.lower().split())
            scores = []
            
            for i, doc in enumerate(documents):
                content = doc.get('content', str(doc))
                doc_words = set(content.lower().split())
                overlap = len(query_words & doc_words)
                score = overlap / len(query_words) if query_words else 0
                scores.append((i, score, doc))
            
            # Sort by score
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-k
            results = []
            for idx, score, doc in scores[:top_k]:
                if score > 0:
                    results.append({
                        'content': doc.get('content', ''),
                        'metadata': doc.get('metadata', {}),
                        'relevance_score': float(score)
                    })
            
            logger.info(f"Keyword search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []
