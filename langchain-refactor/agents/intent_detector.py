"""
Intent Detector
Detects user intent from natural language using keyword/pattern matching
No LLM required - pure local processing
"""

from typing import Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)


class IntentDetector:
    """
    Detects user intent from natural language input.
    Uses keyword/pattern matching - no external API required.
    """
    
    def __init__(
        self,
        model_name: str = None,
        provider: str = "local",
        api_key: Optional[str] = None
    ):
        """
        Initialize intent detector.
        
        Args:
            model_name: Not used (kept for compatibility)
            provider: Always 'local' (kept for compatibility)
            api_key: Not used (kept for compatibility)
        """
        self.provider = "local"
        
        # Intent patterns (keyword-based)
        self.intent_patterns = {
            'scrape': {
                'keywords': ['ambil', 'scrape', 'extract', 'get', 'download', 'fetch', 'grab', 'unduh'],
                'requires_url': True,
                'priority': 5
            },
            'full_pipeline': {
                'keywords': ['analisis lengkap', 'analyze', 'full', 'complete', 'semua', 'all', 'seluruh', 'lengkap', 'pipa', 'pipeline'],
                'requires_url': True,
                'priority': 1
            },
            'preprocess': {
                'keywords': ['bersihkan', 'clean', 'preprocess', 'fitur', 'feature', 'karakteristik'],
                'requires_url': False,
                'priority': 6
            },
            'classify': {
                'keywords': ['klasifikasi', 'classify', 'deteksi', 'detect', 'cek', 'check', 'native ads', 'berita', 'klasifikasikan', 'klasifiaksikan'],
                'requires_url': False,
                'priority': 2
            },
            'retrieve': {
                'keywords': ['cari', 'search', 'contoh', 'example', 'find', 'lookup'],
                'requires_url': False,
                'priority': 7
            },
            'explain': {
                'keywords': ['jelaskan', 'explain', 'kenapa', 'why', 'reasoning', 'alasan', 'detail', 'maksudnya', 'jelasin'],
                'requires_url': False,
                'priority': 3
            },
            'show': {
                'keywords': ['tampilkan', 'lihat', 'mana', 'show', 'view', 'read', 'baca', 'liat', 'tunjukkan'],
                'requires_url': False,
                'priority': 4
            }
        }
        
        logger.info("Intent Detector initialized (local keyword-based)")
    
    def detect(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detect intent from user input using keyword matching.
        
        Args:
            user_input: User's natural language input
            context: Optional conversation context
            
        Returns:
            Intent detection result with extracted entities
        """
        logger.info(f"Detecting intent for: {user_input[:50]}...")
        
        user_lower = user_input.lower()
        url = self._extract_url(user_input)
        
        # Remove URL from keyword search to avoid collisions (e.g., 'ambil' in URL slug)
        search_text = user_lower
        if url:
            search_text = user_lower.replace(url.lower(), "")
        
        # Score each intent
        intent_scores = {}
        for intent, pattern in self.intent_patterns.items():
            score = 0
            
            # Check keywords
            for keyword in pattern['keywords']:
                # Use word boundaries for better accuracy
                if re.search(r'\b' + re.escape(keyword) + r'\b', search_text):
                    # Multi-word keywords get higher score
                    score += len(keyword.split())
                elif keyword in search_text and len(keyword) > 4:
                    # Fallback for substrings only if long enough (e.g. 'klasifikasi' in 'klasifikasikan')
                    score += 1
            
            # Boost score if URL is present and required
            if pattern['requires_url'] and url:
                score += 2
            
            # Penalize if URL required but not present
            if pattern['requires_url'] and not url:
                score -= 1
            
            intent_scores[intent] = score
        
        # Get best intent (sort by score, then by priority for tie-breaks)
        sorted_intents = sorted(
            intent_scores.items(), 
            key=lambda x: (x[1], -self.intent_patterns[x[0]]['priority']), 
            reverse=True
        )
        
        best_intent, best_score = sorted_intents[0]
        
        # If no clear winner, default to chat
        if best_score <= 0:
            return {
                'intent': 'chat',
                'url': '',
                'content': '',
                'query': user_input,
                'confidence': 0.4
            }
        
        # Calculate confidence based on score
        confidence = min(0.5 + (best_score * 0.1), 0.95)

        # Heuristic: If we have a URL and we're asking to classify or explain, 
        # it should probably be a full_pipeline instead of just scrape or individual components
        if url and best_intent == 'scrape':
            if any(k in user_lower for k in self.intent_patterns['classify']['keywords']):
                logger.info("URL + Classify keywords detected. Upgrading intent to full_pipeline.")
                best_intent = 'full_pipeline'
            elif any(k in user_lower for k in self.intent_patterns['explain']['keywords']):
                logger.info("URL + Explain keywords detected. Upgrading intent to full_pipeline.")
                best_intent = 'full_pipeline'

        # Extract entities based on intent
        result = {
            'intent': best_intent,
            'url': url,
            'content': '',
            'query': user_lower,  # Pass the query for downstream logic
            'confidence': confidence
        }
        
        # Intent-specific entity extraction
        if best_intent == 'scrape' or best_intent == 'full_pipeline':
            result['url'] = url
        
        elif best_intent == 'preprocess':
            # Extract content (everything except keywords)
            result['content'] = self._extract_content(user_input, self.intent_patterns['preprocess']['keywords'])
            if not result['content'] and context:
                result['content'] = context.get('last_scraped_content', '')
        
        elif best_intent == 'classify':
            # Extract content for classification
            result['content'] = self._extract_content(user_input, self.intent_patterns['classify']['keywords'])
            if not result['content'] and context:
                result['content'] = context.get('last_scraped_content', '')
        
        elif best_intent == 'retrieve':
            # Extract search query
            result['query'] = self._extract_query(user_input, self.intent_patterns['retrieve']['keywords'])
        
        elif best_intent == 'explain':
            # Use context from previous classification
            if context and context.get('last_classification'):
                result['content'] = context.get('last_scraped_content', '')
        
        logger.info(f"Intent detected: {best_intent} (confidence: {confidence:.2f})")
        return result
    
    def _extract_url(self, text: str) -> str:
        """Extract URL from text using regex."""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(url_pattern, text)
        return match.group(0) if match else ""
    
    def _extract_content(self, text: str, keywords: list) -> str:
        """Extract content by removing intent keywords."""
        # Remove URL
        text_no_url = re.sub(r'https?://[^\s]+', '', text).strip()
        
        # Remove intent keywords
        for keyword in keywords:
            text_no_url = re.sub(r'\b' + re.escape(keyword) + r'\b', '', text_no_url, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        text_no_url = ' '.join(text_no_url.split()).strip()
        
        # Remove common separators
        text_no_url = text_no_url.lstrip(':').strip()
        
        return text_no_url
    
    def _extract_query(self, text: str, keywords: list) -> str:
        """Extract search query by removing intent keywords."""
        query = self._extract_content(text, keywords)
        return query if query else text
