"""
Preprocess Agent Module
Handles data preprocessing, tokenization, and feature extraction.
"""

import logging
import re
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class PreprocessAgent:
    """
    Preprocess Agent that prepares data for both Deep Learning Model
    and Agentic AI components.
    """
    
    def __init__(self, tokenizer_path: Optional[str] = None, max_length: int = 512):
        """
        Initialize the Preprocess Agent.
        
        Args:
            tokenizer_path: Path to tokenizer model (optional)
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.tokenizer = self._load_tokenizer()
        logger.info("Preprocess Agent initialized")
    
    def _load_tokenizer(self):
        """
        Load tokenizer. This is a placeholder - replace with actual tokenizer.
        For example: transformers.AutoTokenizer.from_pretrained()
        """
        # Placeholder - in production, load actual tokenizer
        # from transformers import AutoTokenizer
        # return AutoTokenizer.from_pretrained(self.tokenizer_path)
        return None
    
    def preprocess(self, web_agent_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the data from Web Agent.
        
        Args:
            web_agent_output: Output from Web Agent
            
        Returns:
            Preprocessed data ready for model consumption
        """
        logger.info("Starting preprocessing")
        
        cleaned_input = web_agent_output['cleaned_input']
        
        # Text cleaning
        cleaned_text = self._clean_text(cleaned_input)
        
        # Tokenization
        tokens = self._tokenize(cleaned_text)
        
        # Feature extraction
        features = self._extract_features(cleaned_text)
        
        # Prepare output
        preprocessed_data = {
            'original_input': web_agent_output['raw_input'],
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'features': features,
            'session_id': web_agent_output['session_id'],
            'timestamp': web_agent_output['timestamp'],
            'metadata': {
                **web_agent_output['metadata'],
                'preprocessing_version': '1.0',
                'token_count': len(tokens) if tokens else 0
            }
        }
        
        logger.info(f"Preprocessing complete (tokens: {preprocessed_data['metadata']['token_count']})")
        return preprocessed_data
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (customize as needed)
        # text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        
        # Convert to lowercase (optional)
        # text = text.lower()
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.tokenizer:
            # Use actual tokenizer
            # tokens = self.tokenizer.encode(text, max_length=self.max_length, truncation=True)
            # return tokens
            pass
        
        # Simple word tokenization as fallback
        tokens = text.split()
        
        # Truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        return tokens
    
    def _extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract additional features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'has_question': '?' in text,
            'has_exclamation': '!' in text,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
        
        return features
