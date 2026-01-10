"""
Classification Agent using LangChain
Main agent for classifying content as native ads or pure news
"""

from typing import Dict, Any, Optional
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain.callbacks.manager import CallbackManager
import json
import logging

from prompts.classification_prompts import few_shot_classification_prompt, simple_classification_prompt

logger = logging.getLogger(__name__)


class ClassificationAgent:
    """
    LangChain-based agent for native ads classification.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        provider: str = "openai",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        use_few_shot: bool = True
    ):
        """
        Initialize Classification Agent.
        
        Args:
            model_name: Name of LLM model
            provider: 'openai' or 'huggingface'
            api_key: API key for the provider
            temperature: Sampling temperature
            use_few_shot: Whether to use few-shot learning
        """
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.use_few_shot = use_few_shot
        
        # Initialize LLM
        self.llm = self._initialize_llm(api_key)
        
        # Select prompt
        prompt = few_shot_classification_prompt if use_few_shot else simple_classification_prompt
        
        # Create chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True
        )
        
        logger.info(f"Classification Agent initialized with {model_name} ({provider})")
    
    def _initialize_llm(self, api_key: Optional[str]):
        """Initialize LLM based on provider."""
        if self.provider == "openai":
            return ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                api_key=api_key
            )
        elif self.provider == "openrouter":
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/brianrizqi/agentic-ai-native-ads",
                    "X-Title": "Native Ads Detection System"
                }
            )
        elif self.provider == "huggingface":
            return HuggingFaceEndpoint(
                repo_id=self.model_name,
                temperature=self.temperature,
                huggingfacehub_api_token=api_key
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def classify(
        self,
        content: str,
        title: str = "",
        summary: str = "",
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Classify content as native ads or pure news.
        
        Args:
            content: Main content text
            title: Article title
            summary: Content summary
            context: Retrieved context from RAG
            
        Returns:
            Classification result with label, confidence, and reasoning
        """
        try:
            logger.info("Classifying content...")
            
            # Prepare input
            if self.use_few_shot:
                input_data = {
                    "title": title,
                    "content": content[:800]  # Limit content length
                }
            else:
                input_data = {
                    "title": title,
                    "summary": summary,
                    "content": content[:800],
                    "context": context if context else "No additional context available."
                }
            
            # Run chain
            response = self.chain.run(**input_data)
            
            # Parse response
            result = self._parse_response(response)
            
            # Add metadata
            result['metadata'] = {
                'model': self.model_name,
                'provider': self.provider,
                'use_few_shot': self.use_few_shot,
                'input_length': len(content)
            }
            
            logger.info(f"Classification: {result.get('label')} (confidence: {result.get('confidence', 0):.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return self._get_fallback_classification(content)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        try:
            # Try to extract JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end != -1:
                json_str = response[start:end]
                result = json.loads(json_str)
                
                return {
                    'label': result.get('label', 'unknown'),
                    'confidence': float(result.get('confidence', 0.5)),
                    'reasoning': result.get('reasoning', str(result))
                }
            else:
                raise json.JSONDecodeError("No JSON found", response, 0)
                
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
            
            # Fallback parsing
            import re
            label = 'berita murni'  # Default
            confidence = 0.7
            
            if 'native ads' in response.lower() or 'native advertising' in response.lower():
                label = 'native ads'
                confidence = 0.75
            elif 'berita murni' in response.lower() or 'pure news' in response.lower():
                label = 'berita murni'
                confidence = 0.75
            
            # Try to extract confidence
            conf_match = re.search(r'"confidence":\s*(0\.\d+|1\.0)', response)
            if conf_match:
                confidence = float(conf_match.group(1))
            
            return {
                'label': label,
                'confidence': confidence,
                'reasoning': response[:200]
            }
    
    def _get_fallback_classification(self, text: str) -> Dict[str, Any]:
        """Fallback classification using keywords."""
        text_lower = text.lower()
        
        # Promotional keywords
        promo_keywords = [
            'promo', 'diskon', 'gratis', 'beli', 'dapatkan',
            'penawaran', 'spesial', 'cashback', 'bonus'
        ]
        promo_count = sum(1 for kw in promo_keywords if kw in text_lower)
        
        if promo_count >= 2:
            return {
                'label': 'native ads',
                'confidence': 0.65,
                'reasoning': f'Keyword-based: detected {promo_count} promotional terms'
            }
        else:
            return {
                'label': 'berita murni',
                'confidence': 0.60,
                'reasoning': 'Keyword-based: appears to be pure news content'
            }
