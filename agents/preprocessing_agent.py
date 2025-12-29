"""
Preprocessing Agent Module
Handles data preprocessing, cleaning, and feature extraction.
"""

import logging
import re
import numpy as np
from typing import Dict, Any, List
from collections import Counter

logger = logging.getLogger(__name__)


class PreprocessingAgent:
    """
    Preprocessing Agent yang memproses data dari Web Agent.
    """
    
    def __init__(self, max_length: int = 2000):
        """
        Initialize Preprocessing Agent.
        
        Args:
            max_length: Maximum text length to process
        """
        self.max_length = max_length
        logger.info("Preprocessing Agent initialized")
    
    def process(self, scraped_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process scraped data.
        
        Args:
            scraped_data: Data from Web Agent
            
        Returns:
            Preprocessed and cleaned data
        """
        logger.info(f"Processing data from: {scraped_data.get('url', 'unknown')}")
        
        # Extract raw text
        raw_text = scraped_data.get('text', '')
        
        # Clean text
        cleaned_text = self._clean_text(raw_text)
        
        # Truncate if too long
        if len(cleaned_text) > self.max_length:
            cleaned_text = cleaned_text[:self.max_length]
            logger.info(f"Text truncated to {self.max_length} characters")
        
        # Tokenize
        tokens = self._tokenize(cleaned_text)
        
        # Extract features
        features = self._extract_features(cleaned_text, tokens)
        
        # Extract entities (simple version)
        entities = self._extract_entities(cleaned_text)
        
        # Create summary from paragraphs
        summary = self._create_summary(scraped_data.get('paragraphs', []))
        
        preprocessed_data = {
            'original_url': scraped_data.get('url'),
            'title': scraped_data.get('title'),
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'summary': summary,
            'tokens': tokens,
            'features': features,
            'entities': entities,
            'metadata': {
                **scraped_data.get('metadata', {}),
                'preprocessing_version': '1.0',
                'original_length': len(raw_text),
                'cleaned_length': len(cleaned_text),
                'token_count': len(tokens),
                'truncated': len(raw_text) > self.max_length
            }
        }
        
        logger.info(f"Preprocessing complete: {len(tokens)} tokens, {len(entities)} entities")
        return preprocessed_data
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)
        
        # Remove extra punctuation
        text = re.sub(r'([.,!?;:]){2,}', r'\1', text)
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple word tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _extract_features(self, text: str, tokens: List[str]) -> Dict[str, Any]:
        """
        Extract statistical features from text.
        
        Args:
            text: Cleaned text
            tokens: List of tokens
            
        Returns:
            Dictionary of features
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = text.split()
        word_lengths = [len(word) for word in words]
        
        # Token frequency
        token_freq = Counter(tokens)
        most_common = token_freq.most_common(10)
        
        features = {
            'text_length': len(text),
            'word_count': len(words),
            'unique_word_count': len(set(tokens)),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean(word_lengths) if word_lengths else 0,
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0,
            'most_common_words': most_common,
            'has_question': '?' in text,
            'has_exclamation': '!' in text,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        }
        
        return features
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities (simple pattern-based).
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
            'urls': re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
            'dates': re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text),
            'numbers': re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text),
            'capitalized_words': re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        }
        
        # Limit entities
        for key in entities:
            entities[key] = list(set(entities[key]))[:10]
        
        return entities
    
    def _create_summary(self, paragraphs: List[str], max_sentences: int = 3) -> str:
        """
        Create a simple summary from paragraphs.
        
        Args:
            paragraphs: List of paragraph texts
            max_sentences: Maximum sentences in summary
            
        Returns:
            Summary text
        """
        if not paragraphs:
            return ""
        
        # Take first few paragraphs
        summary_paragraphs = paragraphs[:max_sentences]
        
        # Combine and truncate
        summary = ' '.join(summary_paragraphs)
        
        if len(summary) > 500:
            summary = summary[:500] + '...'
        
        return summary
