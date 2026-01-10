"""
Text Processing Tools for LangChain
Provides tools for text cleaning, tokenization, and feature extraction
"""

from typing import Optional, Type, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForToolRun
import re
import logging
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


class TextCleanerInput(BaseModel):
    """Input schema for TextCleanerTool."""
    text: str = Field(description="Raw text to clean")
    max_length: int = Field(default=2000, description="Maximum text length")


class TextCleanerTool(BaseTool):
    """Tool for cleaning and normalizing text."""
    
    name: str = "text_cleaner"
    description: str = """
    Cleans and normalizes text by removing URLs, emails, extra whitespace,
    and special characters. Returns cleaned text.
    """
    args_schema: Type[BaseModel] = TextCleanerInput
    
    def _run(
        self,
        text: str,
        max_length: int = 2000,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Clean text."""
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove URLs
            text = re.sub(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                '',
                text
            )
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove special characters but keep punctuation
            text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)
            
            # Remove extra punctuation
            text = re.sub(r'([.,!?;:]){2,}', r'\1', text)
            
            # Truncate if too long
            if len(text) > max_length:
                text = text[:max_length]
            
            cleaned = text.strip()
            logger.info(f"Cleaned text: {len(cleaned)} characters")
            return cleaned
            
        except Exception as e:
            logger.error(f"Text cleaning error: {e}")
            return text


class FeatureExtractorInput(BaseModel):
    """Input schema for FeatureExtractorTool."""
    text: str = Field(description="Text to extract features from")


class FeatureExtractorTool(BaseTool):
    """Tool for extracting statistical features from text."""
    
    name: str = "feature_extractor"
    description: str = """
    Extracts statistical features from text including word count, sentence count,
    lexical diversity, and other linguistic features. Returns a dictionary of features.
    """
    args_schema: Type[BaseModel] = FeatureExtractorInput
    return_direct: bool = False
    
    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        """Extract features from text."""
        try:
            # Tokenize
            tokens = re.findall(r'\b\w+\b', text.lower())
            
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Split into words
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
                'avg_word_length': float(np.mean(word_lengths)) if word_lengths else 0,
                'avg_sentence_length': float(np.mean([len(s.split()) for s in sentences])) if sentences else 0,
                'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0,
                'most_common_words': [{'word': word, 'count': count} for word, count in most_common],
                'has_question': '?' in text,
                'has_exclamation': '!' in text,
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
                'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0
            }
            
            logger.info(f"Extracted {len(features)} features")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {}


class EntityExtractorInput(BaseModel):
    """Input schema for EntityExtractorTool."""
    text: str = Field(description="Text to extract entities from")


class EntityExtractorTool(BaseTool):
    """Tool for extracting named entities from text."""
    
    name: str = "entity_extractor"
    description: str = """
    Extracts named entities from text using pattern matching.
    Finds emails, URLs, dates, numbers, and capitalized words.
    Returns a dictionary of entity types and their values.
    """
    args_schema: Type[BaseModel] = EntityExtractorInput
    
    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict[str, List[str]]:
        """Extract entities from text."""
        try:
            entities = {
                'emails': re.findall(
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    text
                ),
                'urls': re.findall(
                    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                    text
                ),
                'dates': re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text),
                'numbers': re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text),
                'capitalized_words': re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            }
            
            # Limit entities and remove duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))[:10]
            
            total_entities = sum(len(v) for v in entities.values())
            logger.info(f"Extracted {total_entities} entities")
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return {}


class SummarizerInput(BaseModel):
    """Input schema for SummarizerTool."""
    paragraphs: List[str] = Field(description="List of paragraphs to summarize")
    max_sentences: int = Field(default=3, description="Maximum sentences in summary")


class SummarizerTool(BaseTool):
    """Tool for creating simple extractive summaries."""
    
    name: str = "text_summarizer"
    description: str = """
    Creates a simple extractive summary from a list of paragraphs.
    Takes the first few paragraphs and combines them into a summary.
    Returns summary text.
    """
    args_schema: Type[BaseModel] = SummarizerInput
    
    def _run(
        self,
        paragraphs: List[str],
        max_sentences: int = 3,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Create summary from paragraphs."""
        try:
            if not paragraphs:
                return ""
            
            # Take first few paragraphs
            summary_paragraphs = paragraphs[:max_sentences]
            
            # Combine
            summary = ' '.join(summary_paragraphs)
            
            # Truncate if too long
            if len(summary) > 500:
                summary = summary[:500] + '...'
            
            logger.info(f"Created summary: {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return ""
