"""
Explanation Agent using LangChain
Generates human-readable explanations for classification decisions
"""

from typing import Dict, Any, Optional
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
import logging

from prompts.classification_prompts import explanation_prompt

logger = logging.getLogger(__name__)


class ExplanationAgent:
    """
    LangChain-based agent for generating explanations.
    Explains WHY content was classified as native ads or pure news.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        provider: str = "openai",
        api_key: Optional[str] = None,
        temperature: float = 0.5
    ):
        """
        Initialize Explanation Agent.
        
        Args:
            model_name: Name of LLM model
            provider: 'openai' or 'huggingface'
            api_key: API key for the provider
            temperature: Sampling temperature (higher = more creative)
        """
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = self._initialize_llm(api_key)
        
        # Create chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=explanation_prompt,
            verbose=True
        )
        
        logger.info(f"Explanation Agent initialized with {model_name} ({provider})")
    
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
    
    def explain(
        self,
        content: str,
        classification_result: Dict[str, Any]
    ) -> str:
        """
        Generate detailed explanation for classification.
        
        Args:
            content: Original content that was classified
            classification_result: Result from ClassificationAgent
            
        Returns:
            Human-readable explanation in Bahasa Indonesia
        """
        try:
            logger.info("Generating explanation...")
            
            # Extract classification details
            label = classification_result.get('label', 'unknown')
            confidence = classification_result.get('confidence', 0)
            reasoning = classification_result.get('reasoning', '')
            
            # Prepare input
            input_data = {
                'content': content[:500],  # Limit length
                'label': label,
                'confidence': f"{confidence:.2f}",
                'reasoning': reasoning
            }
            
            # Run chain
            explanation = self.chain.run(**input_data)
            
            logger.info("Explanation generated successfully")
            return explanation.strip()
            
        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
            return self._get_fallback_explanation(classification_result)
    
    def _get_fallback_explanation(self, classification_result: Dict[str, Any]) -> str:
        """Generate simple fallback explanation."""
        label = classification_result.get('label', 'unknown')
        confidence = classification_result.get('confidence', 0)
        reasoning = classification_result.get('reasoning', 'Tidak ada penjelasan tersedia')
        
        explanation = f"""
Artikel ini diklasifikasikan sebagai: **{label.upper()}**

Tingkat Keyakinan: {confidence:.0%}

Alasan:
{reasoning}

Catatan: Ini adalah penjelasan otomatis berdasarkan analisis AI.
"""
        return explanation.strip()
