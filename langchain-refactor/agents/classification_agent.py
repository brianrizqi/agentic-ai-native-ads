"""
Classification Agent using LangChain
Main agent for classifying content as native ads or pure news
"""

from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import json
import logging
import os

from prompts.classification_prompts import (
    few_shot_classification_prompt, 
    simple_classification_prompt,
    TRAINING_PROMPT_TEMPLATE,
    CONDENSED_TRAINING_PROMPT_TEMPLATE
)

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
        use_few_shot: bool = True,
        lora_path: Optional[str] = None,
        gpu_id: int = 0,
        use_rag: bool = False,
        is_mcq: bool = False
    ):
        """
        Initialize Classification Agent.
        
        Args:
            model_name: Name of LLM model
            provider: 'openai' or 'huggingface'
            api_key: API key for the provider
            temperature: Sampling temperature
            use_few_shot: Whether to use few-shot learning
            lora_path: Path to LoRA adapters (optional)
            gpu_id: GPU ID to use (0, 1, etc.) or -1 for 'auto'
            use_rag: Whether RAG context will be provided
            is_mcq: Force MCQ reasoning-first mode
        """
        self.model_name = model_name
        self.provider = provider
        # Phase 31: Absolute Determinism (forced 0.0 for stability)
        self.temperature = 0.0
        self.use_few_shot = use_few_shot
        self.lora_path = lora_path
        self.gpu_id = gpu_id
        self.use_rag = use_rag
        self.is_mcq = is_mcq  # Phase 46: Decoupled from use_rag to allow JSON restoration
        
        # Phase 19: Model Tiers for scalable RAG
        self.model_tier = self._get_model_tier()
        self.is_small_model = (self.model_tier in ["micro", "small"])
        
        # Initialize LLM
        self.tokenizer = None
        self.llm = self._initialize_llm(api_key)
        
        # Select prompt based on model/provider
        model_name_lower = self.model_name.lower()
        if self.use_rag:
            if self.model_tier == "micro":
                from prompts.classification_prompts import REASONING_MCQ_PROMPT_TEMPLATE
                prompt = PromptTemplate.from_template(REASONING_MCQ_PROMPT_TEMPLATE)
            else:
                prompt = PromptTemplate.from_template(
                    """Klasifikasikan berita berikut sebagai "native ads" atau "berita murni".

Native Ads adalah konten yang MENGGABUNGKAN semua ciri berikut:
1. Nada positif/netral (tidak mengkritik subjek)
2. Bahasa persuasif (mengajak/meyakinkan)
3. Mempromosikan produk/brand/instansi
4. Hanya satu sudut pandang (tidak objektif)

Berita Murni:
- Bisa positif/netral/negatif
- Objektif, menyajikan berbagai sudut pandang
- Tidak mempromosikan produk/brand

{context}

Judul: {title}
Konten: {content}

Output (JSON):
{{"reasoning": "alasan singkat", "label": "native ads" atau "berita murni"}}

Klasifikasi:
"""
                )
        elif self.provider == "local":
            from prompts.classification_prompts import simple_local_prompt
            prompt = simple_local_prompt
        else:
            prompt = few_shot_classification_prompt if use_few_shot else simple_classification_prompt
        
        self.chain = prompt | self.llm | StrOutputParser()
        
        logger.info(f"Classification Agent initialized with {model_name} ({provider})")
    
    def _get_model_tier(self) -> str:
        """Categorize model by scale for prompt optimization."""
        name = self.model_name.lower()
        if any(kw in name for kw in ["270m", "0.5b", "tiny", "micro"]):
            return "micro"
        if any(kw in name for kw in ["1b", "2b", "3b", "small", "lite"]):
            return "small"
        if "qwen" in name and "9b" in name: 
            return "standard"
        return "standard"

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
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                import torch
                
                logger.info(f"Loading local model from: {self.model_name}")
                hf_token = "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"
                
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    token=hf_token,
                    trust_remote_code=True
                )
                
                bnb_config = None
                if torch.cuda.is_available():
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )

                if self.gpu_id == -1:
                    device_map = "auto"
                else:
                    device_map = {"": self.gpu_id} if torch.cuda.is_available() else "auto"

                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32),
                    trust_remote_code=True,
                    token=hf_token
                )
                
                if self.lora_path:
                    logger.info(f"Applying LoRA adapters from: {self.lora_path}")
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(base_model, self.lora_path)
                else:
                    model = base_model
                
                model_name_lower = self.model_name.lower()
                if not tokenizer.chat_template and ("gemma" in model_name_lower or "270" in model_name_lower):
                    tokenizer.chat_template = (
                        "{% for message in messages %}"
                        "{% if message['role'] == 'user' %}"
                        "{{ '<start_of_turn>user\\n' + message['content'] + '<end_of_turn>\\n' }}"
                        "{% elif message['role'] == 'assistant' %}"
                        "{{ '<start_of_turn>model\\n' + message['content'] + '<end_of_turn>\\n' }}"
                        "{% endif %}"
                        "{% endfor %}"
                        "{% if add_generation_prompt %}"
                        "{{ '<start_of_turn>model\\n' }}"
                        "{% endif %}"
                    )

                self.tokenizer = tokenizer
                
                # Phase 34: Universal Stop Sequences (Wait for JSON or EOS)
                stop_sequences = ["}\n", "} \n", "}\n\n", "}", tokenizer.eos_token]
                
                if hasattr(model, 'generation_config'):
                    model.config.max_length = 2048
                    model.generation_config.max_length = 2048
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512, # Phase 34: Standard limit for all local models
                    do_sample=False,
                    repetition_penalty=1.1,
                    return_full_text=False
                )
                
                self.stop_sequences = stop_sequences
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
        context: str = "",
        examples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Classify content as native ads or pure news.
        """
        try:
            logger.info("Classifying content...")
            
            templated_prompt = ""
            input_data = {
                "title": title or content[:100],
                "content": content[:400],
                "context": context
            }
            
            # Phase 34: Ultimate Gold Standard Alignment (Reasoning-First)
            if self.provider == "local" and self.tokenizer:
                from prompts.classification_prompts import ULTIMATE_GOLD_STANDARD_TEMPLATE
                
                # High-Confidence RAG (Threshold: 0.35)
                top_score = examples[0].get('similarity_score', 1.0) if (examples and self.use_rag) else 1.0
                use_this_rag = self.use_rag and examples and top_score > 0.35
                
                rag_context = ""
                if use_this_rag:
                    ex = examples[0]
                    ex_content = ex.get('content', '')[:300].replace('\n', ' ')
                    rag_context = (
                        f"CONTOH RAG (Sim={top_score:.2f}):\n"
                        f"- Konten: {ex_content}...\n"
                        f"- Label: {ex.get('label')}\n"
                    )

                user_msg = ULTIMATE_GOLD_STANDARD_TEMPLATE.format(
                    title=title or content[:60],
                    content=content[:400],
                    context=rag_context
                ).strip()
                
                # Restore original training-fidelity prefix
                prefix_force = '{"reasoning": "' 

                messages = [{"role": "user", "content": user_msg}]
                templated_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                if prefix_force:
                    templated_prompt += prefix_force
                
                stop_seqs = ["}", "\n\n", self.tokenizer.eos_token]
                response = self.llm.invoke(templated_prompt, stop=stop_seqs)
                
                if prefix_force:
                    response = prefix_force + response
                
            else:
                response = self.chain.invoke(input_data)
            
            result = self._parse_response(response)
            
            result['metadata'] = {
                'model': self.model_name,
                'provider': self.provider,
                'use_few_shot': self.use_few_shot,
                'input_length': len(content),
                'raw_prompt': templated_prompt if self.provider == "local" else "",
                'raw_response': response
            }
            
            logger.info(f"Classification: {result.get('label')} (confidence: {result.get('confidence', 0):.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return self._get_fallback_classification(content)
            
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format (Phase 34 Gold Reset)."""
        try:
            import re
            resp_lower = response.lower()
            
            # 1. Primary: JSON Parsing (Training-Fidelity Reset)
            if '{' in response and '}' in response:
                try:
                    json_str = response[response.find('{'):response.rfind('}')+1]
                    json_str = json_str.replace('\n', ' ').strip()
                    data = json.loads(json_str)
                    
                    raw_label = ""
                    if 'label' in data: raw_label = str(data['label']).lower()
                    
                    if raw_label:
                        label = 'native ads' if ("native" in raw_label or "ads" in raw_label or "iklan" in raw_label) else 'berita murni'
                        return {
                            'label': label,
                            'confidence': data.get('confidence', 0.95),
                            'reasoning': data.get('reasoning', 'JSON extraction')
                        }
                except:
                    pass

            # 2. Secondary: Keyword Fallback
            if any(kw in resp_lower for kw in ["native ads", "iklan"]):
                return {'label': 'native ads', 'confidence': 0.85, 'reasoning': 'Keyword match fallback'}
            elif any(kw in resp_lower for kw in ["berita murni", "news"]):
                return {'label': 'berita murni', 'confidence': 0.85, 'reasoning': 'Keyword match fallback'}

            return self._get_fallback_classification(response)
                
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return self._get_fallback_classification(response)
    
    def _get_fallback_classification(self, text: str) -> Dict[str, Any]:
        """Fallback classification using keywords."""
        text_lower = text.lower()
        promo_keywords = ['promo', 'diskon', 'gratis', 'beli', 'dapatkan', 'penawaran', 'spesial', 'iklan']
        promo_count = sum(1 for kw in promo_keywords if kw in text_lower)
        
        if promo_count >= 1:
            return {
                'label': 'native ads',
                'confidence': 0.60,
                'reasoning': 'Keyword fallback'
            }
        else:
            return {
                'label': 'berita murni',
                'confidence': 0.60,
                'reasoning': 'Keyword fallback default'
            }

    def compute_perplexity(self, text: str, prompt: str = "") -> float:
        """Compute perplexity for a given text."""
        if self.provider != "local" or not self.tokenizer:
            return 0.0
        try:
            import torch
            model = self.llm.pipeline.model
            tokenizer = self.tokenizer
            inputs = tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            return perplexity
        except Exception as e:
            logger.error(f"Perplexity calculation error: {e}")
            return 0.0
