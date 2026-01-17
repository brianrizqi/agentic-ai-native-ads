"""
Orchestrator Agent
Main orchestrator that routes user requests to appropriate agents
"""

from typing import Dict, Any, List, Optional
import logging

from agents.intent_detector import IntentDetector
from agents.preprocessing_agent import PreprocessingAgent
from agents.retrieval_agent import RetrievalAgent
from agents.classification_agent import ClassificationAgent
from agents.explanation_agent import ExplanationAgent
from tools import WebScraperTool
from chains.full_pipeline_chain import FullPipelineChain

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Main orchestrator that understands user intent and routes to appropriate agents.
    Maintains conversation memory and handles multi-step workflows.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        provider: str = "openai",
        api_key: Optional[str] = None,
        vectorstore: Optional[Any] = None
    ):
        """
        Initialize orchestrator with all agents.
        
        Args:
            model_name: LLM model name
            provider: LLM provider
            api_key: API key
            vectorstore: Optional vector store for retrieval
        """
        self.model_name = model_name
        self.provider = provider
        self.api_key = api_key
        self.vectorstore = vectorstore
        
        # Initialize intent detector (local, no API needed)
        self.intent_detector = IntentDetector()
        
        # Initialize agents
        self.scraper = WebScraperTool()
        self.preprocessor = PreprocessingAgent()
        self.retriever = RetrievalAgent(vectorstore=vectorstore)
        self.classifier = ClassificationAgent(
            model_name=model_name,
            provider=provider,
            api_key=api_key
        )
        self.explainer = ExplanationAgent(
            model_name=model_name,
            provider=provider,
            api_key=api_key
        )
        self.full_pipeline = FullPipelineChain(
            model_name=model_name,
            provider=provider,
            api_key=api_key,
            vectorstore=vectorstore
        )
        
        # Conversation memory (last 5 messages)
        self.memory: List[Dict[str, str]] = []
        self.max_memory = 5
        
        # Context from previous operations
        self.context = {}
        
        logger.info("Orchestrator Agent initialized with all agents")
    
    def process(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and route to appropriate agent.
        
        Args:
            user_input: User's natural language input
            
        Returns:
            Response from the appropriate agent
        """
        try:
            # Add to memory
            self._add_to_memory("user", user_input)
            
            # Detect intent
            intent_result = self.intent_detector.detect(user_input, context=self.context)
            intent = intent_result['intent']
            
            logger.info(f"🔍 Intent: {intent} (confidence: {intent_result['confidence']:.2f})")
            
            # Route to appropriate agent
            response = self._route_to_agent(intent, intent_result)
            
            # Add response to memory
            self._add_to_memory("assistant", str(response.get('message', '')))
            
            # Update context
            self._update_context(intent, response)
            
            return {
                'intent': intent,
                'confidence': intent_result['confidence'],
                **response
            }
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return {
                'status': 'error',
                'message': f"Maaf, terjadi error: {str(e)}"
            }
    
    def _route_to_agent(self, intent: str, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to appropriate agent based on intent."""
        
        if intent == 'scrape':
            return self._handle_scrape(intent_result)
        
        elif intent == 'preprocess':
            return self._handle_preprocess(intent_result)
        
        elif intent == 'retrieve':
            return self._handle_retrieve(intent_result)
        
        elif intent == 'classify':
            return self._handle_classify(intent_result)
        
        elif intent == 'explain':
            return self._handle_explain(intent_result)
        
        elif intent == 'full_pipeline':
            return self._handle_full_pipeline(intent_result)
        
        elif intent == 'chat':
            return self._handle_chat(intent_result)
        
        else:
            return {
                'status': 'error',
                'message': f"Intent tidak dikenali: {intent}"
            }
    
    def _handle_scrape(self, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle web scraping request."""
        url = intent_result.get('url')
        
        if not url:
            return {
                'status': 'error',
                'message': "URL tidak ditemukan. Mohon berikan URL yang valid."
            }
        
        logger.info(f"🌐 Scraping: {url}")
        result = self.scraper.run({"url": url})
        
        if 'error' in result:
            return {
                'status': 'error',
                'message': f"Gagal scraping: {result['error']}"
            }
        
        # Store in context
        self.context['last_scraped_url'] = url
        self.context['last_scraped_content'] = result.get('text', '')
        self.context['last_scraped_title'] = result.get('title', '')
        self.context['last_scraped_paragraphs'] = result.get('paragraphs', [])
        
        return {
            'status': 'success',
            'message': f"✅ Konten berhasil diambil dari {url}",
            'data': {
                'title': result.get('title', 'N/A'),
                'word_count': len(result.get('text', '').split()),
                'paragraph_count': len(result.get('paragraphs', []))
            }
        }
    
    def _handle_preprocess(self, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text preprocessing request."""
        content = intent_result.get('content') or self.context.get('last_scraped_content', '')
        
        if not content:
            return {
                'status': 'error',
                'message': "Tidak ada teks untuk diproses. Berikan teks atau scrape URL terlebih dahulu."
            }
        
        logger.info("🔧 Preprocessing text...")
        result = self.preprocessor.preprocess(
            text=content,
            paragraphs=self.context.get('last_scraped_paragraphs')
        )
        
        # Store in context
        self.context['last_preprocessed'] = result
        
        return {
            'status': 'success',
            'message': "✅ Preprocessing selesai",
            'data': {
                'original_length': result.get('original_length', 0),
                'cleaned_length': result.get('cleaned_length', 0),
                'features': result.get('features', {}),
                'summary': result.get('summary', '')[:200]
            }
        }
    
    def _handle_retrieve(self, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle retrieval/search request."""
        query = intent_result.get('query') or intent_result.get('content', '')
        
        if not query:
            return {
                'status': 'error',
                'message': "Query tidak ditemukan. Apa yang ingin Anda cari?"
            }
        
        logger.info(f"🔍 Searching: {query[:50]}...")
        results = self.retriever.search_examples(query=query, top_k=5)
        
        if not results:
            return {
                'status': 'success',
                'message': "Tidak ada hasil ditemukan di database.",
                'data': {'results': []}
            }
        
        return {
            'status': 'success',
            'message': f"✅ Ditemukan {len(results)} contoh",
            'data': {
                'results': [
                    {
                        'label': r.get('metadata', {}).get('label', 'unknown'),
                        'content': r.get('content', '')[:150] + '...',
                        'similarity': r.get('score', 0)
                    }
                    for r in results
                ]
            }
        }
    
    def _handle_classify(self, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle classification request."""
        content = intent_result.get('content') or self.context.get('last_scraped_content', '')
        
        if not content:
            return {
                'status': 'error',
                'message': "Tidak ada konten untuk diklasifikasi. Berikan konten atau scrape URL terlebih dahulu."
            }
        
        # Get RAG context if available
        logger.info("🔍 Getting RAG context...")
        rag_context = self.retriever.get_context_for_classification(content)
        
        # Classify
        logger.info("🤖 Classifying content...")
        result = self.classifier.classify(
            content=content,
            title=self.context.get('last_scraped_title', ''),
            summary=self.context.get('last_preprocessed', {}).get('summary', ''),
            context=rag_context
        )
        
        # Store in context
        self.context['last_classification'] = result
        
        return {
            'status': 'success',
            'message': f"✅ Klasifikasi: **{result['label'].upper()}**",
            'data': {
                'label': result['label'],
                'confidence': result['confidence'],
                'reasoning': result.get('reasoning', '')
            }
        }
    
    def _handle_explain(self, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle explanation request."""
        classification = self.context.get('last_classification')
        content = self.context.get('last_scraped_content', '')
        
        if not classification:
            return {
                'status': 'error',
                'message': "Belum ada klasifikasi. Klasifikasikan konten terlebih dahulu."
            }
        
        logger.info("📝 Generating explanation...")
        explanation = self.explainer.explain(
            content=content,
            classification_result=classification
        )
        
        return {
            'status': 'success',
            'message': "✅ Penjelasan detail:",
            'data': {
                'explanation': explanation
            }
        }
    
    def _handle_full_pipeline(self, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle full pipeline request."""
        url = intent_result.get('url')
        
        if not url:
            return {
                'status': 'error',
                'message': "URL tidak ditemukan untuk analisis lengkap."
            }
        
        logger.info(f"🚀 Running full pipeline: {url}")
        result = self.full_pipeline.run(url)
        
        if result.get('status') == 'failed':
            return {
                'status': 'error',
                'message': f"Pipeline gagal: {result.get('error', 'Unknown error')}"
            }
        
        # Store in context
        self.context['last_classification'] = result.get('classification', {})
        
        return {
            'status': 'success',
            'message': "✅ Analisis lengkap selesai",
            'data': {
                'title': result.get('title', ''),
                'classification': result.get('classification', {}),
                'explanation': result.get('explanation', ''),
                'features': result.get('features', {})
            }
        }
    
    def _handle_chat(self, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general chat/questions."""
        query = intent_result.get('query', '')
        
        # Simple FAQ responses
        query_lower = query.lower()
        
        if 'native ads' in query_lower and any(word in query_lower for word in ['apa', 'what', 'pengertian']):
            return {
                'status': 'success',
                'message': """Native ads adalah konten iklan yang menyerupai konten editorial/berita.
                
Karakteristik:
1. Nada positif/netral (tidak mengkritik)
2. Bahasa persuasif
3. Mempromosikan produk/brand
4. Satu sudut pandang

Saya bisa membantu Anda mengklasifikasikan konten. Berikan URL atau teks!"""
            }
        
        if any(word in query_lower for word in ['help', 'bantuan', 'bisa apa']):
            return {
                'status': 'success',
                'message': """Saya bisa membantu Anda:
                
1. 🌐 **Scraping** - Ambil konten dari URL
2. 🔧 **Preprocessing** - Bersihkan & analisis teks
3. 🔍 **Retrieval** - Cari contoh di database
4. 🤖 **Classification** - Klasifikasi native ads/berita murni
5. 📝 **Explanation** - Jelaskan hasil klasifikasi
6. 🚀 **Full Pipeline** - Analisis lengkap

Contoh: "analisis url https://example.com" """
            }
        
        return {
            'status': 'success',
            'message': f"Saya tidak yakin bagaimana menjawab itu. Ketik 'help' untuk melihat apa yang bisa saya lakukan!"
        }
    
    def _add_to_memory(self, role: str, content: str):
        """Add message to conversation memory."""
        self.memory.append({"role": role, "content": content})
        
        # Keep only last N messages
        if len(self.memory) > self.max_memory * 2:  # *2 for user+assistant pairs
            self.memory = self.memory[-(self.max_memory * 2):]
    
    def _update_context(self, intent: str, response: Dict[str, Any]):
        """Update context based on response."""
        if response.get('status') == 'success':
            self.context['last_intent'] = intent
            self.context['last_response'] = response
    
    def get_memory(self) -> List[Dict[str, str]]:
        """Get conversation memory."""
        return self.memory
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory = []
        self.context = {}
        logger.info("Memory and context cleared")
