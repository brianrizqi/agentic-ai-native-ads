"""
Retrieval Agent
Searches vector database for similar examples to provide RAG context
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Agent for retrieving similar examples from vector database.
    Provides RAG context for classification.
    """
    
    def __init__(self, vectorstore: Optional[Any] = None):
        """
        Initialize retrieval agent.
        
        Args:
            vectorstore: Vector store instance (optional)
        """
        self.vectorstore = vectorstore
        logger.info(f"Retrieval Agent initialized (vectorstore: {vectorstore is not None})")
    
    def get_context_for_classification(
        self,
        text: str,
        top_k: int = 3
    ) -> str:
        """
        Get RAG context for classification.
        Retrieves similar examples to help classifier.
        
        Args:
            text: Text to find similar examples for
            top_k: Number of examples to retrieve
            
        Returns:
            Formatted context string
        """
        if not self.vectorstore:
            logger.warning("No vectorstore available, returning empty context")
            return ""
        
        try:
            from tools.retrieval_tools import VectorSearchTool
            
            retriever = VectorSearchTool(vectorstore=self.vectorstore)
            results = retriever.run({"query": text[:500], "top_k": top_k})
            
            if not results:
                return ""
            
            # Format context
            context_lines = ["Berikut adalah beberapa contoh artikel serupa sebagai referensi (Beberapa mungkin memiliki label yang berbeda, gunakan sebagai pembanding):"]
            for i, doc in enumerate(results[:top_k], 1):
                label = doc.get('metadata', {}).get('label', 'unknown')
                reasoning = doc.get('metadata', {}).get('reasoning', 'Tidak tersedia')
                content = doc.get('content', '')[:200]
                
                context_lines.append(f"\n[CONTOH {i}]")
                context_lines.append(f"Label: {label}")
                context_lines.append(f"Alasan: {reasoning}")
                context_lines.append(f"Isi Artikel: {content}...")
            
            context = "\n".join(context_lines)
            logger.info(f"Retrieved {len(results)} examples for RAG context")
            return context
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return ""
    
    def search_examples(
        self,
        query: str,
        label: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar examples.
        
        Args:
            query: Search query
            label: Filter by label ('native ads' or 'berita murni')
            top_k: Number of results
            
        Returns:
            List of matching examples
        """
        if not self.vectorstore:
            logger.warning("No vectorstore available")
            return []
        
        try:
            from tools.retrieval_tools import VectorSearchTool
            
            retriever = VectorSearchTool(vectorstore=self.vectorstore)
            results = retriever.run({"query": query, "top_k": top_k * 2})  # Get more for filtering
            
            # Filter by label if specified
            if label and results:
                results = [
                    r for r in results
                    if r.get('metadata', {}).get('label', '').lower() == label.lower()
                ]
            
            # Limit to top_k
            results = results[:top_k]
            
            logger.info(f"Found {len(results)} examples for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def search_by_topic(
        self,
        topic: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search examples by topic/keyword.
        
        Args:
            topic: Topic or keyword
            top_k: Number of results
            
        Returns:
            List of matching examples
        """
        return self.search_examples(query=topic, top_k=top_k)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Statistics about the vector database
        """
        if not self.vectorstore:
            return {
                "available": False,
                "message": "No vectorstore configured"
            }
        
        try:
            # Try to get collection info
            stats = {
                "available": True,
                "vectorstore_type": type(self.vectorstore).__name__
            }
            
            # Try to count documents (if supported)
            try:
                if hasattr(self.vectorstore, '_collection'):
                    count = self.vectorstore._collection.count()
                    stats['document_count'] = count
            except:
                pass
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics error: {e}")
            return {
                "available": True,
                "error": str(e)
            }
