"""
Explanation Agent Module
Generates human-readable explanations for classifications.
Supports: OpenAI, HuggingFace (Local)
"""

import logging
import json
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ExplanationAgent:
    """
    Explanation Agent yang menghasilkan penjelasan untuk klasifikasi.
    """
    
    def __init__(self, api_key: str = '', model_name: str = 'openai/gpt-oss-20b',
                 provider: str = 'huggingface', temperature: float = 0.7, 
                 max_tokens: int = 1500):
        """
        Initialize Explanation Agent.
        
        Args:
            api_key: API key (optional for local)
            model_name: LLM model name
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
        logger.info(f"Explanation Agent initialized with {model_name} ({provider})")
    
    def _initialize_client(self):
        """Initialize LLM client."""
        if self.provider == 'openai':
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                return client
            except ImportError:
                logger.warning("OpenAI library not installed")
                return None
                
                return None
                
        elif self.provider == 'huggingface':
            try:
                from transformers import pipeline
                import torch
                
                print(f"   [INFO] Loading HuggingFace model for explanation: {self.model_name}...")
                
                # Use device_map="auto" for MIG compatibility
                # Force framework='pt' to avoid Keras 3 issue
                generator = pipeline(
                    "text-generation", 
                    model=self.model_name, 
                    device_map="auto",
                    torch_dtype=torch.float16,
                    framework="pt",  # Force PyTorch
                    token=self.api_key
                )
                print(f"   [SUCCESS] Explanation model loaded!")
                return generator
            except ImportError as e:
                print(f"   [ERROR] Missing dependencies: {e}")
                logger.error(f"transformers/torch not installed: {e}")
                raise e
            except Exception as e:
                print(f"   [ERROR] Failed to load HF model: {e}")
                logger.error(f"Failed to load HF model: {e}")
                raise e
                
        return None
    
    def explain(self, classification: Dict[str, Any], 
                preprocessed_data: Dict[str, Any],
                retrieved_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate explanation for classification.
        """
        logger.info(f"Generating explanation with {self.provider}")
        
        # Extract data
        text = preprocessed_data.get('cleaned_text', '')[:500]
        title = preprocessed_data.get('title', '')
        
        # Build prompt
        prompt = self._build_explanation_prompt(
            text, title, classification, retrieved_context
        )
        
        # Generate explanation
        explanation_text = ""
        
        try:
            if self.provider == 'openai' and self.client:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert at explaining AI classifications."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                explanation_text = response.choices[0].message.content
                
            elif self.provider == 'huggingface' and self.client:
                response = self.client(
                    prompt,
                    max_new_tokens=512,
                    num_return_sequences=1,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    truncation=True,
                    pad_token_id=self.client.tokenizer.eos_token_id
                )
                explanation_text = response[0]['generated_text']
                if explanation_text.startswith(prompt):
                    explanation_text = explanation_text[len(prompt):]
            
            else:
                return self._get_fallback_explanation(classification)
                
            # Structure result
            explanation = self._structure_explanation(explanation_text, classification)
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
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
        
        prompt = f"""[INST] Buatkan penjelasan yang jelas untuk hasil klasifikasi ini dalam Bahasa Indonesia.

JUDUL KONTEN: {title}

KUTIPAN KONTEN:
{text}

HASIL KLASIFIKASI:
- Label: {label}
- Confidence: {confidence:.0%}
- Reasoning: {reasoning}

JUMLAH DOKUMEN KONTEKS YANG DIGUNAKAN: {len(context)}

Berikan penjelasan dalam Bahasa Indonesia yang mencakup:
1. Ringkasan konten yang diklasifikasikan
2. Mengapa klasifikasi ini dipilih
3. Faktor-faktor kunci yang mempengaruhi keputusan
4. Penjelasan tingkat confidence
5. Interpretasi alternatif (jika ada)

Tulis dengan nada yang jelas dan profesional dalam Bahasa Indonesia. [/INST]"""
        
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
                'provider': self.provider,
                'explanation_length': len(explanation_text)
            }
        }
    
    def _extract_summary(self, text: str) -> str:
        """Extract summary from explanation."""
        paragraphs = text.strip().split('\n\n')
        if paragraphs:
            return paragraphs[0][:300] + '...'
        return text[:300] + '...'
    
    def _extract_key_factors(self, text: str) -> List[str]:
        """Extract key factors from explanation."""
        factors = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if any(line.startswith(p) for p in ['1.', '2.', '3.', '-', '*', '•']):
                cleaned = line.lstrip('0123456789.-*• ')
                if len(cleaned) > 10:  # Minimum length check
                    factors.append(cleaned)
        return factors[:5]
    
    def _extract_alternatives(self, text: str) -> List[str]:
        """Extract alternative interpretations."""
        alternatives = []
        lines = text.split('\n')
        for line in lines:
            if any(k in line.lower() for k in ['alternative', 'however', 'could also']):
                alternatives.append(line.strip())
        return alternatives[:3]
    
    def _get_fallback_explanation(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback explanation."""
        label = classification.get('label', 'unknown')
        confidence = classification.get('confidence', 0)
        
        return {
            'summary': f"Classified as '{label}' with {confidence:.0%} confidence.",
            'detailed_explanation': "Explanation generation failed or unavailable.",
            'key_factors': [],
            'classification_label': label,
            'confidence_score': confidence,
            'alternative_interpretations': [],
            'metadata': {'model': 'fallback'}
        }
