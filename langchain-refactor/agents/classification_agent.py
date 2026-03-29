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
        """
        self.model_name = model_name
        self.provider = provider
        # Phase 31: Absolute Determinism (forced 0.0 for stability)
        self.temperature = 0.0
        self.use_few_shot = use_few_shot
        self.lora_path = lora_path
        self.gpu_id = gpu_id
        self.use_rag = use_rag
        self.is_mcq = is_mcq
        
        # Phase 19: Model Tiers for scalable RAG
        self.model_tier = self._get_model_tier()
        
        # Initialize LLM
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
        if "qwen" in name and "9b" in name: 
            return "standard"
        return "standard"

    def _initialize_llm(self, api_key: Optional[str]):
        """Initialize LLM based on provider."""
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
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                import torch
                
                logger.info(f"Loading local model: {self.model_name}")
                hf_token = "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"
                
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token, trust_remote_code=True)
                
                if self.gpu_id == -1:
                    device_map = "auto"
                else:
                    device_map = {"": self.gpu_id} if torch.cuda.is_available() else "auto"

                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=device_map,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32),
                    load_in_4bit=True if torch.cuda.is_available() else False,
                    trust_remote_code=True,
                    token=hf_token
                )
                
                if self.lora_path:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(base_model, self.lora_path)
                else:
                    model = base_model
                
                self.tokenizer = tokenizer
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    do_sample=False,
                    repetition_penalty=1.1,
                    return_full_text=False
                )
                
                return HuggingFacePipeline(pipeline=pipe)

            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                raise ValueError(f"Could not load local model: {e}")
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
            
            # Phase 35: Dynamic Prompt Complexity (The Low-PPL Calibrator)
            if self.provider == "local" and self.tokenizer:
                from prompts.classification_prompts import ULTIMATE_GOLD_STANDARD_TEMPLATE, ZERO_SHOT_GOLD_STANDARD_TEMPLATE
                
                # High-Confidence RAG (Threshold: 0.35)
                top_score = examples[0].get('similarity_score', 1.0) if (examples and self.use_rag) else 1.0
                use_this_rag = self.use_rag and examples and top_score > 0.35
                
                rag_context = ""
                if use_this_rag:
                    ex = examples[0]
                    # Micro Tiers get shorter RAG strings to prevent overload
                    limit = 100 if self.model_tier == "micro" else 300
                    ex_content = ex.get('content', '')[:limit].replace('\n', ' ')
                    rag_context = f"CONTOH RAG (Sim={top_score:.2f}):\n- Konten: {ex_content}...\n- Label: {ex.get('label')}\n"

                # TIER SELECTION: Zero-Shot for Micro, Two-Shot for Standard
                if self.model_tier == "micro":
                    template = ZERO_SHOT_GOLD_STANDARD_TEMPLATE
                else:
                    template = ULTIMATE_GOLD_STANDARD_TEMPLATE

                user_msg = template.format(
                    title=title or content[:60],
                    content=content[:400],
                    context=rag_context
                ).strip()
                
                prefix_force = '{"reasoning": "' 
                messages = [{"role": "user", "content": user_msg}]
                templated_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                if prefix_force:
                    templated_prompt += prefix_force
                
                stop_seqs = ["}", "\n\n", self.tokenizer.eos_token]
                response = self.llm.invoke(templated_prompt, stop=stop_seqs)
                
                if prefix_force:
                    response = prefix_force + response
                
            else:
                input_data = {"title": title or content[:100], "content": content[:400], "context": context}
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
            return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': f'Error fallback: {str(e)}'}
            
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format (Phase 34/35 Gold Standards)."""
        try:
            import re
            resp_lower = response.lower()
            
            # 1. Primary: JSON Parsing
            if '{' in response and '}' in response:
                try:
                    json_str = response[response.find('{'):response.rfind('}')+1]
                    json_str = json_str.replace('\n', ' ').strip()
                    data = json.loads(json_str)
                    
                    raw_label = str(data.get('label', '')).lower()
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

            return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': 'Ambiguous fallback'}
                
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': 'Parse error fallback'}

    def compute_perplexity(self, text: str, prompt: str = "") -> float:
        """Compute perplexity for a given text, conditioned on a prompt (Phase 35: Generation-Only PPL)."""
        if self.provider != "local" or not self.tokenizer:
            return 0.0
        try:
            import torch
            model = self.llm.pipeline.model
            tokenizer = self.tokenizer
            
            if prompt:
                # PHASE 35: Targeted PPL (only measures generation entropy)
                full_text = prompt + text
                encoding = tokenizer(full_text, return_tensors="pt")
                input_ids = encoding["input_ids"].to(model.device)
                
                # Find where the prompt ends
                prompt_encoding = tokenizer(prompt, return_tensors="pt")
                prompt_len = prompt_encoding["input_ids"].shape[1]
                
                # Mask out the prompt tokens in labels so loss is only calculated on target text
                labels = input_ids.clone()
                labels[:, :prompt_len] = -100
                
                with torch.no_grad():
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss
                    return torch.exp(loss).item()
            else:
                inputs = tokenizer(text, return_tensors="pt")
                input_ids = inputs["input_ids"].to(model.device)
                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)
                    return torch.exp(outputs.loss).item()
        except Exception as e:
            logger.error(f"Perplexity calculation error: {e}")
            return 0.0
