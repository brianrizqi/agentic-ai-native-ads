"""
LLM Classifier Agent Module
Uses Large Language Model for classification with context.
Supports: OpenAI, Ollama (Local), HuggingFace (Local)
"""

import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class LLMClassifierAgent:
    """
    LLM Classifier Agent yang menggunakan LLM untuk klasifikasi.
    """
    
    def __init__(self, api_key: str = '', model_name: str = 'llama2', 
                 provider: str = 'ollama', temperature: float = 0.3, 
                 max_tokens: int = 1000):
        """
        Initialize LLM Classifier Agent.
        
        Args:
            api_key: API key for remote service (optional for local)
            model_name: LLM model name (e.g. 'llama2', 'mistral', 'gpt-4')
            provider: 'openai', 'ollama', or 'huggingface'
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = self._initialize_client()
        logger.info(f"LLM Classifier initialized with {model_name} ({provider})")
    
    def _initialize_client(self):
        """Initialize LLM client based on provider."""
        if self.provider == 'openai':
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                return client
            except ImportError:
                logger.warning("OpenAI library not installed")
                return None
                
        elif self.provider == 'ollama':
            try:
                import ollama
                # Check connection
                try:
                    ollama.list()
                    return ollama
                except Exception:
                    logger.warning("Ollama service not running or not accessible")
                    return None
            except ImportError:
                logger.warning("ollama library not installed (pip install ollama)")
                return None
                
        elif self.provider == 'huggingface':
            try:
                from transformers import pipeline
                import torch
                
                device = 0 if torch.cuda.is_available() else -1
                device_name = "GPU" if device == 0 else "CPU"
                logger.info(f"Loading local HuggingFace model: {self.model_name} on {device_name}...")
                
                # Use text-generation pipeline with device
                generator = pipeline("text-generation", model=self.model_name, device=device)
                return generator
            except ImportError:
                logger.warning("transformers/torch not installed")
                return None
            except Exception as e:
                logger.error(f"Failed to load HF model: {e}")
                return None
                
        return None
    
    def classify(self, preprocessed_data: Dict[str, Any], 
                 retrieved_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Classify input using LLM with retrieved context.
        """
        logger.info(f"Classifying with {self.provider}")
        
        # Extract data
        text = preprocessed_data.get('cleaned_text', '')
        title = preprocessed_data.get('title', '')
        summary = preprocessed_data.get('summary', '')
        
        # Format context
        context_text = self._format_context(retrieved_context)
        
        # Build prompt
        prompt = self._build_classification_prompt(text, title, summary, context_text)
        
        # Call LLM
        response_text = ""
        
        try:
            if self.provider == 'openai' and self.client:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert classifier."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                response_text = response.choices[0].message.content
                
            elif self.provider == 'ollama' and self.client:
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': self.temperature,
                        'num_predict': self.max_tokens
                    }
                )
                response_text = response['response']
                
            elif self.provider == 'huggingface' and self.client:
                # HF pipeline
                response = self.client(
                    prompt, 
                    max_length=len(prompt.split()) + self.max_tokens,
                    num_return_sequences=1,
                    temperature=self.temperature
                )
                response_text = response[0]['generated_text']
                # Strip prompt from response if duplicated
                if response_text.startswith(prompt):
                    response_text = response_text[len(prompt):]
            
            else:
                logger.warning("No valid client available, using fallback")
                return self._get_fallback_classification(text)
                
            # Parse result
            classification = self._parse_llm_response(response_text)
            
        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            classification = self._get_fallback_classification(text)
        
        # Add metadata
        classification['metadata'] = {
            'model': self.model_name,
            'provider': self.provider,
            'context_documents': len(retrieved_context),
            'input_length': len(text)
        }
        
        logger.info(f"Classification: {classification.get('label')} (confidence: {classification.get('confidence', 0):.2f})")
        return classification
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format retrieved context for prompt."""
        if not context:
            return "No additional context available."
        
        formatted = []
        for i, doc in enumerate(context, 1):
            content = doc.get('content', '')[:300]
            # Handle dictionary content (if content is dict)
            if isinstance(content, dict):
                content = str(content)
                
            score = doc.get('relevance_score', 0)
            formatted.append(f"[Context {i}] (Relevance: {score:.2f})\n{content}")
        
        return "\n\n".join(formatted)
    
    def _build_classification_prompt(self, text: str, title: str, 
                                     summary: str, context: str) -> str:
        """Build classification prompt."""
        prompt = f"""[INST] You are an expert classifier. Analyze the following content and classify it.

Title: {title}
Summary: {summary}

Content Excerpt:
{text[:800]}

Related Examples (Context):
{context}

Task: Classify this content into ONE of these categories:
- Native Advertising (Paid content looking like news)
- Editorial Content (Pure news/journalism)
- Sponsored Content (Clearly labeled paid content)

Provide output in valid JSON format:
{{
  "label": "category_name",
  "confidence": 0.0_to_1.0,
  "reasoning": "brief explanation"
}}
[/INST]"""
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        try:
            # Clean up response (find first '{' and last '}')
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            
            if start != -1 and end != -1:
                json_str = response_text[start:end]
                result = json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON found", response_text, 0)
            
            # Ensure fields
            return {
                'label': result.get('label', 'unknown'),
                'confidence': float(result.get('confidence', 0.5)),
                'reasoning': result.get('reasoning', str(result))
            }
            
        except Exception:
            # Fallback parsing
            logger.warning("Failed to parse JSON response")
            return {
                'label': 'uncertain',
                'confidence': 0.0,
                'reasoning': response_text[:200]
            }
    
    def _get_fallback_classification(self, text: str) -> Dict[str, Any]:
        """Fallback when LLM fails."""
        return {
            'label': 'unknown',
            'confidence': 0.0,
            'reasoning': 'LLM inference failed, fallback used'
        }
