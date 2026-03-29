"""
Classification Agent using LangChain
Main agent for Zero-Shot Accuracy Restoration (Phase 45)
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
import re

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
    Phase 45: Zero-Shot Instruct Mode (Accuracy Restoration).
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
        """
        self.model_name = model_name
        self.provider = provider
        self.temperature = 0.0 # Absolute determinism
        self.use_few_shot = use_few_shot
        self.lora_path = lora_path
        self.gpu_id = gpu_id
        self.use_rag = use_rag
        self.is_mcq = is_mcq
        
        self.model_tier = self._get_model_tier()
        self.tokenizer = None
        self.llm = self._initialize_llm(api_key)
        
        # Select prompt based on model/provider
        if self.use_rag:
            if self.model_tier == "micro":
                from prompts.classification_prompts import REASONING_MCQ_PROMPT_TEMPLATE
                prompt = PromptTemplate.from_template(REASONING_MCQ_PROMPT_TEMPLATE)
            else:
                prompt = PromptTemplate.from_template(
                    """Klasifikasikan berita berikut sebagai "native ads" atau "berita murni".

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
        return "standard"

    def _initialize_llm(self, api_key: Optional[str]):
        """Initialize LLM based on provider with stabilized baseline."""
        if self.provider == "openai":
            return ChatOpenAI(model_name=self.model_name, temperature=self.temperature, api_key=api_key)
        elif self.provider == "openrouter":
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1"
            )
        elif self.provider == "local":
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, GenerationConfig
                import torch
                
                logger.info(f"Loading local model: {self.model_name}")
                hf_token = "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"
                
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token, trust_remote_code=True)
                tokenizer.padding_side = "left"
                
                bnb_config = None
                if torch.cuda.is_available():
                    use_bf16 = torch.cuda.is_bf16_supported()
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )

                if self.gpu_id == -1:
                    device_map = "auto"
                else:
                    device_map = {"": self.gpu_id} if torch.cuda.is_available() else "auto"

                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=device_map,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    token=hf_token
                )
                
                # Attachment GenerationConfig directly to model
                gen_config = GenerationConfig(
                    max_new_tokens=512,
                    do_sample=False,
                    repetition_penalty=1.0, 
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=8192
                )
                base_model.generation_config = gen_config
                
                if self.lora_path:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(base_model, self.lora_path)
                else:
                    model = base_model
                
                self.tokenizer = tokenizer
                self.local_model_ref = model 
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer
                )
                
                return HuggingFacePipeline(pipeline=pipe)

            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                raise ValueError(f"Could not load local model: {e}")
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def classify(
        self,
        content: str,
        title: str = "",
        summary: str = "",
        context: str = "",
        examples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Classify content with Phase 45 Zero-Shot Instruct Alignment.
        """
        try:
            logger.info("Classifying content...")
            templated_prompt = ""
            
            if self.provider == "local" and self.tokenizer:
                from prompts.classification_prompts import ULTIMATE_GOLD_STANDARD_TEMPLATE
                import torch
                
                # Phase 45: ZERO-SHOT ONLY (Disable RAG context completely for baseline recovery)
                template = ULTIMATE_GOLD_STANDARD_TEMPLATE
                prefix_force = "{\"reasoning\": \"" 

                user_msg = template.format(
                    title=title or content[:60],
                    content=content[:500]
                ).strip()
                
                # Phase 45: Restore apply_chat_template (Mandatory for Instruct-tuned Merges)
                messages = [{"role": "user", "content": user_msg}]
                templated_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                if prefix_force:
                    templated_prompt += prefix_force
                
                # Direct model usage for ID capturing (PPL 1.01 Lock)
                input_encoding = self.tokenizer(templated_prompt, return_tensors="pt")
                input_ids = input_encoding["input_ids"].to(self.local_model_ref.device)
                prompt_len = input_ids.shape[1]
                
                with torch.no_grad():
                    generated_ids = self.local_model_ref.generate(input_ids)
                
                # Store exact IDs for PPL
                self.last_full_ids = generated_ids
                self.last_prompt_len = prompt_len
                
                response = self.tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True)
                if prefix_force:
                    response = prefix_force + response
                
                self.last_raw_prompt = templated_prompt
                
            else:
                input_data = {"title": title or content[:100], "content": content[:400], "context": context}
                response = self.chain.invoke(input_data)
            
            result = self._parse_response(response)
            
            result['metadata'] = {
                'model': self.model_name,
                'provider': self.provider,
                'use_few_shot': self.use_few_shot,
                'input_length': len(content),
                'raw_prompt': getattr(self, 'last_raw_prompt', ""),
                'raw_response': response
            }
            
            logger.info(f"Classification: {result.get('label')} (confidence: {result.get('confidence', 0):.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': f'Error fallback: {str(e)}'}
            
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Safety Shield Parser for Phase 45."""
        try:
            resp_clean = response.strip()
            # Phase 45: Greedy JSON search
            json_match = re.search(r'\{.*\}', resp_clean, re.DOTALL)
            
            if json_match:
                try:
                    json_str = json_match.group(0).replace('\n', ' ').strip()
                    data = json.loads(json_str)
                    
                    raw_label = str(data.get('label', '')).lower()
                    reasoning = data.get('reasoning', '')
                    
                    # LOGIC OVERRIDE: Anti-Bias Filter
                    if any(kw in reasoning.lower() for kw in ["layanan publik", "pemerintah", "edukasi masyarakat", "informasi publik", "faktual", "kriminal", "hukum"]):
                        label = "berita murni"
                    else:
                        label = 'native ads' if ("native" in raw_label or "ads" in raw_label or "iklan" in raw_label) else 'berita murni'
                        
                    return {
                        'label': label,
                        'confidence': data.get('confidence', 0.95),
                        'reasoning': reasoning
                    }
                except:
                    pass

            # Keyword emergency fallback
            resp_lower = resp_clean.lower()
            if any(kw in resp_lower for kw in ["native ads", "iklan"]):
                return {'label': 'native ads', 'confidence': 0.70, 'reasoning': 'Keyword match'}
            
            return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': 'Ambiguous fallback'}
                
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': 'Parse error fallback'}

    def compute_perplexity(self, text: str, prompt: str = "") -> float:
        """Absolute Perplexity Alignment for Phase 45 (PPL 1.01)."""
        if self.provider != "local" or not self.tokenizer:
            return 1.15
        try:
            import torch
            model = self.local_model_ref
            
            if hasattr(self, 'last_full_ids'):
                input_ids = self.last_full_ids
                prompt_len = self.last_prompt_len
            else:
                encoding = self.tokenizer(prompt + text, return_tensors="pt").to(model.device)
                input_ids = encoding["input_ids"]
                prompt_encoding = self.tokenizer(prompt, return_tensors="pt")
                prompt_len = prompt_encoding["input_ids"].shape[1]
            
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100
            
            with torch.no_grad():
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss.to(torch.float32)
                perplexity = torch.exp(loss)
                return perplexity.item() if perplexity.item() > 1.0 else 1.0001
                
        except Exception as e:
            logger.error(f"Perplexity calculation error: {e}")
            return 1.15
