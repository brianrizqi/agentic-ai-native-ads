"""
Preprocessing Agent
Handles text cleaning, feature extraction, and summarization
"""

from typing import Dict, Any, List, Optional
import logging

from tools import TextCleanerTool, FeatureExtractorTool, SummarizerTool

logger = logging.getLogger(__name__)


class PreprocessingAgent:
    """
    Agent for preprocessing text content.
    Provides text cleaning, feature extraction, and summarization.
    """
    
    def __init__(self):
        """Initialize preprocessing tools."""
        self.text_cleaner = TextCleanerTool()
        self.feature_extractor = FeatureExtractorTool()
        self.summarizer = SummarizerTool()
        
        logger.info("Preprocessing Agent initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean raw text (remove HTML, special chars, normalize whitespace).
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        try:
            cleaned = self.text_cleaner.run({"text": text})
            logger.info(f"Text cleaned: {len(text)} → {len(cleaned)} chars")
            return cleaned
        except Exception as e:
            logger.error(f"Text cleaning error: {e}")
            return text
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract features from text (word count, readability, etc).
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of features
        """
        try:
            features = self.feature_extractor.run({"text": text})
            logger.info(f"Features extracted: {list(features.keys())}")
            return features
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {}
    
    def summarize(self, text: str = None, paragraphs: List[str] = None, max_length: int = 200) -> str:
        """
        Create summary from text or paragraphs.
        
        Args:
            text: Full text to summarize (optional)
            paragraphs: List of paragraphs (optional)
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        try:
            if paragraphs:
                summary = self.summarizer.run({"paragraphs": paragraphs})
            elif text:
                # Split into paragraphs
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                summary = self.summarizer.run({"paragraphs": paragraphs})
            else:
                return ""
            
            # Truncate if needed
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            
            logger.info(f"Summary created: {len(summary)} chars")
            return summary
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return ""
    
    def preprocess(
        self,
        text: str,
        tasks: List[str] = None,
        paragraphs: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run multiple preprocessing tasks.
        
        Args:
            text: Raw text to process
            tasks: List of tasks to run ['clean', 'features', 'summarize']
                  If None, runs all tasks
            paragraphs: Optional list of paragraphs for summarization
            
        Returns:
            Dictionary with results from requested tasks
        """
        if tasks is None:
            tasks = ['clean', 'features', 'summarize']
        
        results = {
            'original_text': text,
            'original_length': len(text)
        }
        
        logger.info(f"Running preprocessing tasks: {tasks}")
        
        # Clean text
        if 'clean' in tasks:
            cleaned_text = self.clean_text(text)
            results['cleaned_text'] = cleaned_text
            results['cleaned_length'] = len(cleaned_text)
        else:
            cleaned_text = text
        
        # Extract features
        if 'features' in tasks:
            features = self.extract_features(cleaned_text)
            results['features'] = features
        
        # Summarize
        if 'summarize' in tasks:
            summary = self.summarize(
                text=cleaned_text if not paragraphs else None,
                paragraphs=paragraphs
            )
            results['summary'] = summary
        
        logger.info(f"Preprocessing complete: {len(results)} result fields")
        return results
    
    def process_for_classification(self, text: str, paragraphs: List[str] = None) -> Dict[str, Any]:
        """
        Preprocess text specifically for classification pipeline.
        Returns cleaned text, features, and summary.
        
        Args:
            text: Raw text
            paragraphs: Optional paragraphs
            
        Returns:
            Preprocessed data ready for classification
        """
        return self.preprocess(
            text=text,
            tasks=['clean', 'features', 'summarize'],
            paragraphs=paragraphs
        )
