"""
LLM Classifier Agent Module
Uses Large Language Model for classification with context.
"""

import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class LLMClassifierAgent:
    """
    LLM Classifier Agent yang menggunakan LLM untuk klasifikasi.
    """
    
    def __init__(self, api_key: str, model_name: str = 'gpt-4', 
                 temperature: float = 0.3, max_tokens: int = 1000):
        """
        Initialize LLM Classifier Agent.
        
        Args:
            api_key: API key for LLM service
            model_name: LLM model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = self._initialize_client()
        logger.info(f"LLM Classifier initialized with {model_name}")
    
    def _initialize_client(self):
        """Initialize LLM client."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialized")
            return client
        except ImportError:
            logger.warning("OpenAI library not installed, using placeholder")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
            return None
    
    def classify(self, preprocessed_data: Dict[str, Any], 
                 retrieved_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Classify input using LLM with retrieved context.
        
        Args:
            preprocessed_data: Data from Preprocessing Agent
            retrieved_context: Context from Retriever Agent
            
        Returns:
            Classification result with reasoning
        """
        logger.info("Classifying with LLM")
        
        # Extract data
        text = preprocessed_data.get('cleaned_text', '')
        title = preprocessed_data.get('title', '')
        summary = preprocessed_data.get('summary', '')
        
        # Format context
        context_text = self._format_context(retrieved_context)
        
        # Build prompt
        prompt = self._build_classification_prompt(text, title, summary, context_text)
        
        # Call LLM
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert classifier. Analyze the content and provide structured classification."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                result_text = response.choices[0].message.content
                classification = self._parse_llm_response(result_text)
                
            except Exception as e:
                logger.error(f"LLM API call failed: {e}")
                classification = self._get_fallback_classification(text)
        else:
            classification = self._get_fallback_classification(text)
        
        # Add metadata
        classification['metadata'] = {
            'model': self.model_name,
            'context_documents': len(retrieved_context),
            'input_length': len(text),
            'title': title
        }
        
        logger.info(f"Classification: {classification.get('label')} (confidence: {classification.get('confidence', 0):.2f})")
        return classification
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format retrieved context for prompt."""
        if not context:
            return "No additional context available."
        
        formatted = []
        for i, doc in enumerate(context, 1):
            content = doc.get('content', '')[:300]  # Limit length
            score = doc.get('relevance_score', 0)
            formatted.append(f"[Context {i}] (Relevance: {score:.2f})\n{content}")
        
        return "\n\n".join(formatted)
    
    def _build_classification_prompt(self, text: str, title: str, 
                                     summary: str, context: str) -> str:
        """Build classification prompt."""
        prompt = f"""Analyze and classify the following content:

TITLE: {title}

SUMMARY: {summary}

FULL TEXT (excerpt):
{text[:1000]}

RETRIEVED CONTEXT:
{context}

Please provide a classification with the following information:
1. Primary category/label
2. Confidence score (0-1)
3. Brief reasoning
4. Key topics identified
5. Sentiment (positive/neutral/negative)

Respond in JSON format with keys: label, confidence, reasoning, topics, sentiment"""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        try:
            # Try to parse as JSON
            result = json.loads(response_text)
            
            # Ensure required fields
            if 'label' not in result:
                result['label'] = 'unknown'
            if 'confidence' not in result:
                result['confidence'] = 0.5
            if 'reasoning' not in result:
                result['reasoning'] = response_text
            
            return result
            
        except json.JSONDecodeError:
            # Fallback parsing
            logger.warning("Failed to parse JSON response, using text extraction")
            
            return {
                'label': 'informational',
                'confidence': 0.7,
                'reasoning': response_text,
                'topics': [],
                'sentiment': 'neutral'
            }
    
    def _get_fallback_classification(self, text: str) -> Dict[str, Any]:
        """Fallback classification when LLM is unavailable."""
        logger.warning("Using fallback classification")
        
        # Simple keyword-based classification
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['buy', 'purchase', 'price', 'order']):
            label = 'transactional'
        elif any(word in text_lower for word in ['how', 'what', 'why', 'when', 'where']):
            label = 'informational'
        elif any(word in text_lower for word in ['login', 'account', 'profile', 'settings']):
            label = 'navigational'
        else:
            label = 'general'
        
        return {
            'label': label,
            'confidence': 0.6,
            'reasoning': 'Fallback keyword-based classification',
            'topics': [],
            'sentiment': 'neutral'
        }
