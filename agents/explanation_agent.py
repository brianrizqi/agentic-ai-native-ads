"""
Explanation Agent Module
Generates human-readable explanations for classifications.
"""

import logging
import json
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ExplanationAgent:
    """
    Explanation Agent yang menghasilkan penjelasan untuk klasifikasi.
    """
    
    def __init__(self, api_key: str, model_name: str = 'gpt-4',
                 temperature: float = 0.7, max_tokens: int = 1500):
        """
        Initialize Explanation Agent.
        
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
        logger.info(f"Explanation Agent initialized with {model_name}")
    
    def _initialize_client(self):
        """Initialize LLM client."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            return client
        except ImportError:
            logger.warning("OpenAI library not installed")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize client: {e}")
            return None
    
    def explain(self, classification: Dict[str, Any], 
                preprocessed_data: Dict[str, Any],
                retrieved_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate explanation for classification.
        
        Args:
            classification: Classification result
            preprocessed_data: Preprocessed input data
            retrieved_context: Retrieved context documents
            
        Returns:
            Detailed explanation
        """
        logger.info("Generating explanation")
        
        # Extract data
        text = preprocessed_data.get('cleaned_text', '')[:500]
        title = preprocessed_data.get('title', '')
        
        # Build prompt
        prompt = self._build_explanation_prompt(
            text, title, classification, retrieved_context
        )
        
        # Generate explanation
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert at explaining AI classifications in clear, understandable language."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                explanation_text = response.choices[0].message.content
                explanation = self._structure_explanation(explanation_text, classification)
                
            except Exception as e:
                logger.error(f"Explanation generation failed: {e}")
                explanation = self._get_fallback_explanation(classification)
        else:
            explanation = self._get_fallback_explanation(classification)
        
        logger.info("Explanation generated successfully")
        return explanation
    
    def _build_explanation_prompt(self, text: str, title: str,
                                   classification: Dict[str, Any],
                                   context: List[Dict[str, Any]]) -> str:
        """Build explanation prompt."""
        label = classification.get('label', 'unknown')
        confidence = classification.get('confidence', 0)
        reasoning = classification.get('reasoning', '')
        
        prompt = f"""Generate a clear, comprehensive explanation for the following classification:

CONTENT TITLE: {title}

CONTENT EXCERPT:
{text}

CLASSIFICATION RESULT:
- Label: {label}
- Confidence: {confidence:.0%}
- Reasoning: {reasoning}

NUMBER OF CONTEXT DOCUMENTS USED: {len(context)}

Please provide:
1. A brief summary of what was classified
2. Why this classification was chosen
3. Key factors that influenced the decision
4. Confidence level explanation
5. Any alternative interpretations considered

Write in a clear, professional tone suitable for a research report."""
        
        return prompt
    
    def _structure_explanation(self, explanation_text: str, 
                               classification: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the explanation into components."""
        return {
            'summary': self._extract_summary(explanation_text),
            'detailed_explanation': explanation_text,
            'key_factors': self._extract_key_factors(explanation_text),
            'classification_label': classification.get('label'),
            'confidence_score': classification.get('confidence'),
            'alternative_interpretations': self._extract_alternatives(explanation_text),
            'metadata': {
                'model': self.model_name,
                'explanation_length': len(explanation_text),
                'temperature': self.temperature
            }
        }
    
    def _extract_summary(self, text: str) -> str:
        """Extract summary from explanation."""
        # Simple extraction: first paragraph or first 200 chars
        paragraphs = text.split('\n\n')
        if paragraphs:
            summary = paragraphs[0]
            if len(summary) > 200:
                summary = summary[:200] + '...'
            return summary
        return text[:200] + '...'
    
    def _extract_key_factors(self, text: str) -> List[str]:
        """Extract key factors from explanation."""
        # Look for numbered or bulleted lists
        factors = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Check for numbered items (1., 2., etc.) or bullets (-, *, •)
            if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•']):
                # Clean up the line
                cleaned = line.lstrip('0123456789.-*• ')
                if cleaned:
                    factors.append(cleaned)
        
        return factors[:5]  # Return top 5 factors
    
    def _extract_alternatives(self, text: str) -> List[str]:
        """Extract alternative interpretations."""
        # Simple keyword-based extraction
        alternatives = []
        text_lower = text.lower()
        
        if 'alternative' in text_lower or 'however' in text_lower or 'could also' in text_lower:
            # Extract sentences containing these keywords
            sentences = text.split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in ['alternative', 'however', 'could also', 'might be']):
                    alternatives.append(sentence.strip())
        
        return alternatives[:3]
    
    def _get_fallback_explanation(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback explanation."""
        label = classification.get('label', 'unknown')
        confidence = classification.get('confidence', 0)
        reasoning = classification.get('reasoning', 'No reasoning provided')
        
        summary = f"The content was classified as '{label}' with {confidence:.0%} confidence."
        
        detailed = f"""Classification Explanation:

The content has been classified as '{label}' based on the analysis performed.

Confidence Level: {confidence:.0%}
This confidence score indicates {'high' if confidence > 0.8 else 'moderate' if confidence > 0.5 else 'low'} certainty in the classification.

Reasoning:
{reasoning}

Key Factors:
- Content analysis and pattern matching
- Contextual relevance assessment
- Semantic similarity evaluation

This classification was generated using automated analysis. For critical decisions, human review is recommended."""
        
        return {
            'summary': summary,
            'detailed_explanation': detailed,
            'key_factors': [
                'Content analysis',
                'Pattern matching',
                'Contextual relevance'
            ],
            'classification_label': label,
            'confidence_score': confidence,
            'alternative_interpretations': [],
            'metadata': {
                'model': 'fallback',
                'explanation_length': len(detailed)
            }
        }
