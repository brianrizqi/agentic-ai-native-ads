"""
Explanation Agent using LangChain
Generates human-readable explanations for classification decisions
"""

from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
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
        temperature: float = 0.5,
        llm: Optional[Any] = None
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
        
        # Use provided LLM or initialize new one
        if llm:
            self.llm = llm
            logger.info(f"Explanation Agent using provided LLM instance")
        else:
            self.llm = self._initialize_llm(api_key)
        
        # Create chain using LCEL
        self.chain = explanation_prompt | self.llm | StrOutputParser()
        
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
        elif self.provider == "local":
            # Load local fine-tuned model
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                
                logger.info(f"Loading local model from: {self.model_name}")
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype="auto"
                )
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    temperature=self.temperature,
                    do_sample=True
                )
                
                return HuggingFacePipeline(pipeline=pipe)
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                raise ValueError(f"Could not load local model from {self.model_name}: {e}")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def explain(
        self,
        content: str,
        classification_result: Dict[str, Any],
        title: str = ""
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
                'content': content[:1500],  # Increased length for more context
                'label': label,
                'confidence': f"{confidence:.2f}",
                'reasoning': reasoning,
                'title': title or "Tanpa Judul"
            }
            
            # Run chain using LCEL invoke
            explanation = self.chain.invoke(input_data)
            # AGGRESSIVE CLEANING: Strip variety of possible training leaks
            # 1. Remove JSON blocks or system-like headers
            leak_markers = [
                "Output JSON", "Output (JSON)", "Format JSON", "Klasifikasi Output", 
                "Klasifikasi:", "Penjelasan:", "Native Ads adalah", "Ciri-ciri Native Ads", 
                "Output JSON:", "Result:", "Label:", "Confidence:"
            ]
            for marker in leak_markers:
                if marker in explanation:
                    # Keep text BEFORE the marker if it's long enough, otherwise keep text AFTER if it contains analysis
                    parts = explanation.split(marker)
                    if len(parts[0].strip()) > 50:
                        explanation = parts[0].strip()
                    else:
                        explanation = " ".join(parts[1:]).strip()
            
            # 2. Remove actual JSON structures using regex
            import re
            explanation = re.sub(r'\{.*\}', '', explanation, flags=re.DOTALL).strip()
            explanation = re.sub(r'```json.*```', '', explanation, flags=re.DOTALL).strip()
            explanation = re.sub(r'```.*```', '', explanation, flags=re.DOTALL).strip()
            
            # 3. Final cleanup of trailing colons or partial headers
            explanation = re.sub(r'(Klasifikasi|Penjelasan|Output)\s*:\s*$', '', explanation, flags=re.IGNORECASE).strip()
            
            logger.info("Explanation generated and aggressively cleaned")
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
