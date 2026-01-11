"""
Classification Agent using LangChain
Main agent for classifying content as native ads or pure news
"""

from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.output_parsers import StrOutputParser
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
        
        # Select prompt based on provider
        if self.provider == "local":
            # Use simplified prompt for local models
            from prompts.classification_prompts import simple_local_prompt
            prompt = simple_local_prompt
        else:
            # Use few-shot or simple prompt for API models
            prompt = few_shot_classification_prompt if use_few_shot else simple_classification_prompt
        
        # Create chain using LCEL (more robust than deprecated LLMChain)
        self.chain = prompt | self.llm | StrOutputParser()
        
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
        elif self.provider == "local":
            # Load local fine-tuned model (e.g., Llama with LoRA)
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
                    max_new_tokens=150,  # Reduced to prevent duplicates
                    temperature=0.1,  # Lower temperature for consistency
                    do_sample=False,  # Greedy decoding to avoid duplicates
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                    return_full_text=False  # Only return generated text, not prompt
                )
                
                return HuggingFacePipeline(pipeline=pipe)
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                raise ValueError(f"Could not load local model from {self.model_name}: {e}")
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
            
            # Run chain using LCEL invoke
            response = self.chain.invoke(input_data)
            
            # Parse response (response is already a string thanks to StrOutputParser)
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
            # Clean up response - remove prompt echo if present
            if "Klasifikasi:" in response:
                response = response.split("Klasifikasi:")[-1].strip()
            if "Output (JSON):" in response:
                response = response.split("Output (JSON):")[-1].strip()
            
            # Try to extract JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                
                return {
                    'label': result.get('label', 'unknown'),
                    'confidence': float(result.get('confidence', 0.5)),
                    'reasoning': result.get('reasoning', str(result))[:200]  # Truncate long reasoning
                }
            else:
                raise json.JSONDecodeError("No JSON found", response, 0)
                
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
            logger.info(f"Raw response (first 500 chars): {response[:500]}")
            
            # If multiple JSON objects, extract only the first one
            if response.count('{') > 1:
                try:
                    first_json_start = response.find('{')
                    first_json_end = response.find('}', first_json_start) + 1
                    response = response[first_json_start:first_json_end]
                    # Try parsing again
                    result = json.loads(response)
                    logger.info(f"Successfully extracted first JSON from multiple outputs")
                    return {
                        'label': result.get('label', 'unknown'),
                        'confidence': float(result.get('confidence', 0.5)),
                        'reasoning': result.get('reasoning', str(result))[:200]
                    }
                except:
                    pass
            
            # Improved fallback parsing with keyword analysis
            import re
            response_lower = response.lower()
            
            # Count indicators for each class
            native_ads_indicators = [
                'native ads', 'native advertising', 'advertorial', 'sponsored',
                'promosi', 'iklan', 'brand', 'produk', 'layanan',
                'persuasif', 'mengajak', 'meyakinkan'
            ]
            
            berita_murni_indicators = [
                'berita murni', 'pure news', 'objektif',
                'berbagai sudut pandang', 'kritik', 'investigasi'
            ]
            
            native_score = sum(1 for indicator in native_ads_indicators if indicator in response_lower)
            berita_score = sum(1 for indicator in berita_murni_indicators if indicator in response_lower)
            
            logger.info(f"Keyword analysis - Native ads: {native_score}, Berita murni: {berita_score}")
            
            # Determine label based on scores
            if native_score > berita_score:
                label = 'native ads'
                confidence = min(0.6 + (native_score * 0.05), 0.85)
            elif berita_score > native_score:
                label = 'berita murni'
                confidence = min(0.6 + (berita_score * 0.05), 0.85)
            else:
                # If tied, check for explicit mentions
                if 'native ads' in response_lower or 'advertorial' in response_lower:
                    label = 'native ads'
                    confidence = 0.65
                else:
                    label = 'berita murni'
                    confidence = 0.5
            
            # Try to extract confidence from response
            conf_match = re.search(r'"confidence":\s*(0\.\d+|1\.0)', response)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                except:
                    pass
            
            logger.info(f"Fallback result: {label} (confidence: {confidence:.2f})")
            
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
