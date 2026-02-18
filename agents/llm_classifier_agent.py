"""
LLM Classifier Agent Module
Uses Large Language Model for classification with context.
Supports: OpenAI, HuggingFace (Local)
"""

import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class LLMClassifierAgent:
    """
    LLM Classifier Agent yang menggunakan LLM untuk klasifikasi.
    """
    
    def __init__(self, api_key: str = '', model_name: str = 'openai/gpt-oss-20b', 
                 provider: str = 'huggingface', temperature: float = 0.3, 
                 max_tokens: int = 1000, lora_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize LLM Classifier Agent.
        
        Args:
            api_key: API key for remote service (optional for local)
            model_name: LLM model name (e.g. 'llama2', 'mistral', 'gpt-4')
            provider: 'openai', 'ollama', or 'huggingface'
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            lora_path: Path to LoRA adapters (optional)
            device: Device to use (e.g. 'cpu', 'cuda', 'auto')
        """
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.lora_path = lora_path
        self.device = device or 'auto'
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
                
                return None
        
        elif self.provider == 'huggingface':
            try:
                from transformers import pipeline
                import torch
                
                print(f"   [INFO] Loading HuggingFace model: {self.model_name}...")
                print(f"   [INFO] Using device_map='auto' for MIG compatibility...")
                
                if self.lora_path:
                    print(f"   [INFO] Loading LoRA adapters from: {self.lora_path}...")
                    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                    from peft import PeftModel
                    
                    # BNB only works on CUDA
                    if self.device != 'cpu':
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                        )
                    else:
                        bnb_config = None
                    
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.api_key)
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=bnb_config,
                        device_map=self.device if self.device != 'cpu' else None,
                        torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
                        token=self.api_key
                    )
                    if self.device == 'cpu':
                        base_model = base_model.to('cpu')
                        
                    model = PeftModel.from_pretrained(base_model, self.lora_path)
                    
                    generator = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        framework="pt",
                        device='cpu' if self.device == 'cpu' else None
                    )
                else:
                    # Use device_map="auto" for MIG compatibility (no sudo needed)
                    # Force framework='pt' to avoid Keras 3 issue
                    generator = pipeline(
                        "text-generation", 
                        model=self.model_name, 
                        device_map=self.device if self.device != 'cpu' else None,
                        device=0 if self.device == 'cuda' else (-1 if self.device == 'cpu' else None),
                        torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
                        framework="pt",  # Force PyTorch (avoid TensorFlow/Keras)
                        token=self.api_key
                    )
                print(f"   [SUCCESS] Model loaded successfully on {self.device}!")
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
                
            elif self.provider == 'huggingface' and self.client:
                # HF pipeline
                response = self.client(
                    prompt, 
                    max_new_tokens=512,  # Increased for complete JSON response
                    num_return_sequences=1,
                    temperature=self.temperature if self.temperature > 0 else 0.7,
                    do_sample=True,  # Always sample to avoid repetition
                    top_p=0.95,  # Nucleus sampling
                    repetition_penalty=1.2,  # Prevent repetitive text
                    truncation=True,
                    pad_token_id=self.client.tokenizer.eos_token_id
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
        """Build classification prompt with BALANCED examples."""
        prompt = f"""[INST] You are an expert classifier for detecting native advertising in Indonesian news.

CRITICAL INSTRUCTIONS:
- Analyze carefully - NOT everything is native ads!
- Berita murni (pure news) is EQUALLY common as native ads
- Look for CLEAR promotional intent before labeling as native ads
- When in doubt, prefer "berita murni"

Title: {title}
Summary: {summary}

Kutipan Konten:
{text[:800]}

Contoh Relevan (Konteks):
{context}

CONTOH KLASIFIKASI (BALANCED):

CONTOH 1 - NATIVE ADS:
"Promo spesial BRI di HUT ke-128, diskon hingga Rp1.28 juta untuk berbagai produk perbankan."
Label: native ads
Alasan: Promosi produk bank, diskon, ajakan transaksi.

CONTOH 2 - BERITA MURNI:
"Presiden Jokowi menunjuk Ridwan Kamil sebagai kurator infrastruktur IKN pada hari Selasa."
Label: berita murni
Alasan: Berita faktual kebijakan pemerintah, tidak ada promosi produk.

CONTOH 3 - NATIVE ADS:
"Dapatkan cashback hingga 50% untuk pembelian produk elektronik di Tokopedia hari ini!"
Label: native ads
Alasan: Ajakan beli dengan cashback, promosi e-commerce.

CONTOH 4 - BERITA MURNI:
"Inflasi Indonesia pada bulan Maret tercatat 4.97%, turun dari bulan sebelumnya menurut BPS."
Label: berita murni
Alasan: Laporan ekonomi faktual dari sumber resmi, tidak ada ajakan beli.

CONTOH 5 - NATIVE ADS:
"Investasi emas Antam kini lebih mudah dengan aplikasi baru, gratis biaya admin bulan ini."
Label: native ads
Alasan: Promosi aplikasi investasi, gratis biaya, ajakan menggunakan produk.

CONTOH 6 - BERITA MURNI:
"Mahkamah Agung menolak kasasi terdakwa kasus korupsi dengan vonis 8 tahun penjara."
Label: berita murni
Alasan: Berita hukum faktual, tidak ada promosi atau ajakan komersial.

TUGAS: Klasifikasikan konten di atas ke dalam SALAH SATU kategori:
- "native ads" → Iklan berbayar yang menyerupai berita, ada promosi/ajakan beli
- "berita murni" → Konten jurnalistik murni, laporan faktual tanpa promosi

OUTPUT FORMAT (JSON):
{{
  "label": "native ads" atau "berita murni",
  "confidence": 0.0-1.0,
  "reasoning": "penjelasan singkat dalam Bahasa Indonesia"
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
            
        except Exception as e:
            # Fallback parsing - extract from text
            logger.warning(f"Failed to parse JSON: {e}")
            
            import re
            label = 'berita murni'  # Default
            confidence = 0.7
            
            # Extract label from text
            if 'native ads' in response_text.lower() or 'native advertising' in response_text.lower():
                label = 'native ads'
                confidence = 0.75
            elif 'berita murni' in response_text.lower() or 'editorial' in response_text.lower():
                label = 'berita murni'
                confidence = 0.75
            
            # Try to extract confidence
            conf_match = re.search(r'"confidence":\s*(0\.\d+|1\.0)', response_text)
            if conf_match:
                confidence = float(conf_match.group(1))
            
            return {
                'label': label,
                'confidence': confidence,
                'reasoning': response_text[:200]
            }
    
    def _get_fallback_classification(self, text: str) -> Dict[str, Any]:
        """Fallback when LLM fails - keyword-based classification."""
        text_lower = text.lower()
        
        # Check for promotional keywords
        promo_keywords = ['promo', 'diskon', 'gratis', 'beli', 'dapatkan', 'penawaran', 'spesial', 'cashback']
        promo_count = sum(1 for kw in promo_keywords if kw in text_lower)
        
        if promo_count >= 2:
            return {
                'label': 'native ads',
                'confidence': 0.65,
                'reasoning': f'Berbasis kata kunci: terdeteksi {promo_count} istilah promosi'
            }
        else:
            return {
                'label': 'berita murni',
                'confidence': 0.60,
                'reasoning': 'Berbasis kata kunci: tampak seperti konten berita murni'
            }
