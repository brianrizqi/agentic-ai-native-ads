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
        top_k: int = 3,
        max_reasoning_length: Optional[int] = None,
        minimalist: bool = False
    ) -> str:
        """
        Get RAG context for classification.
        Retrieves similar examples to help classifier.
        
        Args:
            text: Text to find similar examples for
            top_k: Number of examples to retrieve
            max_reasoning_length: Optional truncation length for reasoning metadata
            minimalist: If True, use ultra-compact format for smaller models (Phase 18)
            
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
            if minimalist:
                # Phase 18: Ultra-minimalist Few-Shot format
                context_lines = ["\n[REFERENSI CONTOH]"]
                for i, doc in enumerate(results[:top_k], 1):
                    label = doc.get('metadata', {}).get('label', 'unknown')
                    content = doc.get('content', '')[:150].replace('\n', ' ')
                    context_lines.append(f"- Konten: {content} -> Label: {label}")
            else:
                # Standard Expert RAG format
                context_lines = ["Berikut adalah beberapa contoh artikel serupa sebagai referensi:"]
                for i, doc in enumerate(results[:top_k], 1):
                    label = doc.get('metadata', {}).get('label', 'unknown')
                    reasoning = doc.get('metadata', {}).get('reasoning', 'Tidak tersedia')
                    
                    # Truncate reasoning if requested
                    if max_reasoning_length and len(reasoning) > max_reasoning_length:
                        reasoning = reasoning[:max_reasoning_length].strip() + "..."
                    
                    content = doc.get('content', '')[:200]
                    
                    context_lines.append(f"\n[CONTOH {i}]")
                    context_lines.append(f"Label: {label}")
                    context_lines.append(f"Alasan: {reasoning}")
                    context_lines.append(f"Isi Artikel: {content}...")
            
            context = "\n".join(context_lines)
            logger.info(f"Retrieved {len(results)} examples for RAG context (minimalist: {minimalist})")
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

    def get_examples_for_history(
        self,
        text: str,
        top_k: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get similar examples formatted for conversation history (Phase 20).
        
        Args:
            text: Query text
            top_k: Number of examples
            
        Returns:
            List of example dicts with 'content', 'label', 'reasoning'
        """
        if not self.vectorstore:
            return []
            
        try:
            from tools.retrieval_tools import VectorSearchTool
            retriever = VectorSearchTool(vectorstore=self.vectorstore)
            results = retriever.run({"query": text[:500], "top_k": top_k})
            
            examples = []
            for doc_dict in results: # VectorSearchTool returns list of dicts with 'content', 'metadata', 'similarity_score'
                examples.append({
                    "content": doc_dict.get('content', ''),
                    "label": doc_dict.get('metadata', {}).get('label', 'unknown'),
                    "reasoning": doc_dict.get('metadata', {}).get('reasoning', ''),
                    "similarity_score": doc_dict.get('similarity_score', 0.0)
                })
            return examples
        except Exception as e:
            logger.error(f"Error getting examples for history: {e}")
            return []
    
    def _search_by_label(
        self,
        text: str,
        label: str,
        k: int
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k most similar examples filtered to a specific label.

        Excludes any doc that is (near-)identical to the query — if the test article
        also lives in the vectorstore, returning it as its own 'reference' would leak
        the ground-truth label. Over-fetch then drop self-matches before taking top-k.
        """
        if not self.vectorstore:
            return []
        try:
            results = self.vectorstore.similarity_search_with_score(
                text[:500], k=k * 4, filter_label=label
            )
            query_key = text.strip()[:200]
            examples = []
            for doc, score in results:
                if doc.page_content.strip()[:200] == query_key:
                    continue  # leakage guard: skip the query article itself
                examples.append({
                    "content": doc.page_content,
                    "label": doc.metadata.get("label", label),
                    "reasoning": doc.metadata.get("reasoning", ""),
                    "similarity_score": float(score),
                })
                if len(examples) >= k:
                    break
            return examples
        except Exception as e:
            logger.error(f"Label-filtered search error ({label}): {e}")
            return []

    def get_balanced_examples_for_history(
        self,
        text: str,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Get balanced RAG examples: top_k/2 native ads + top_k/2 berita murni.
        Interleaved so the model sees contrast between both classes.
        Prevents the retrieval from being dominated by one class and biasing predictions.
        """
        k_per_class = max(1, (top_k + 1) // 2)
        native_examples = self._search_by_label(text, "native ads", k_per_class)
        news_examples = self._search_by_label(text, "berita murni", k_per_class)

        # Interleave: ads, news, ads, news — so the model always sees contrast
        combined = []
        for i in range(max(len(native_examples), len(news_examples))):
            if i < len(native_examples):
                combined.append(native_examples[i])
            if i < len(news_examples):
                combined.append(news_examples[i])
        return combined[:top_k]

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
