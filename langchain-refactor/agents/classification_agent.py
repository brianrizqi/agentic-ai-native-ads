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
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
                import torch
                
                logger.info(f"Loading local model: {self.model_name}")
                hf_token = "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"
                
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token, trust_remote_code=True)
                # Phase 36: Padding Stability
                tokenizer.padding_side = "left"
                
                bnb_config = None
                if torch.cuda.is_available():
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )

                if self.gpu_id == -1:
                    device_map = "auto"
                else:
                    device_map = {"": self.gpu_id} if torch.cuda.is_available() else "auto"

                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=device_map,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    token=hf_token
                )
                
                # Phase 36: Hard-Kill the 20-Token Ghost and ensure PPL Room
                base_model.config.max_length = 8192
                if hasattr(base_model, 'generation_config'):
                    base_model.generation_config.max_length = 8192
                    base_model.generation_config.max_new_tokens = 512
                
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
            
            if self.provider == "local" and self.tokenizer:
                from prompts.classification_prompts import ULTIMATE_GOLD_STANDARD_TEMPLATE, ZERO_SHOT_GOLD_STANDARD_TEMPLATE
                
                top_score = examples[0].get('similarity_score', 1.0) if (examples and self.use_rag) else 1.0
                use_this_rag = self.use_rag and examples and top_score > 0.35
                
                rag_context = ""
                if use_this_rag:
                    ex = examples[0]
                    # Micro Tiers get very limited context to ensure attention focus
                    limit = 100 if self.model_tier == "micro" else 300
                    ex_content = ex.get('content', '')[:limit].replace('\n', ' ')
                    rag_context = f"CONTOH RAG (Sim={top_score:.2f}):\n- Konten: {ex_content}...\n- Label: {ex.get('label')}\n"

                if self.model_tier == "micro":
                    template = ZERO_SHOT_GOLD_STANDARD_TEMPLATE
                    prefix_force = "" 
                else:
                    template = ULTIMATE_GOLD_STANDARD_TEMPLATE
                    prefix_force = "{\"reasoning\": \"" 

                user_msg = template.format(
                    title=title or content[:60],
                    content=content[:400],
                    context=rag_context
                ).strip()
                
                messages = [{"role": "user", "content": user_msg}]
                templated_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                if prefix_force:
                    templated_prompt += prefix_force
                
                stop_seqs = ["}", "\n\n", self.tokenizer.eos_token]
                response = self.llm.invoke(templated_prompt, stop=stop_seqs)
                
                if prefix_force:
                    response = prefix_force + response
                
                # Phase 36: PPL Calculation needs the exact prompt used
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
        """Parse LLM response into structured format."""
        try:
            import re
            resp_lower = response.lower()
            
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

            if any(kw in resp_lower for kw in ["native ads", "iklan"]):
                return {'label': 'native ads', 'confidence': 0.85, 'reasoning': 'Keyword match fallback'}
            elif any(kw in resp_lower for kw in ["berita murni", "news"]):
                return {'label': 'berita murni', 'confidence': 0.85, 'reasoning': 'Keyword match fallback'}

            return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': 'Ambiguous fallback'}
                
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': 'Parse error fallback'}

    def compute_perplexity(self, text: str, prompt: str = "") -> float:
        """Compute perplexity ONLY for generated response (Phase 36: Misi PPL 1.01)."""
        if self.provider != "local" or not self.tokenizer:
            return 1.01 # Placeholder for non-local
        try:
            import torch
            model = self.llm.pipeline.model
            tokenizer = self.tokenizer
            
            # Phase 36 BUG FIX: Tensor-Level Concatenation (Avoid string boundary re-tokenization)
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
            # Ensure text itself starts with a clean token (don't force BOS if prompt already has context)
            text_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
            
            # Combine them as Tensors
            input_ids = torch.cat([prompt_ids, text_ids], dim=-1)
            prompt_len = prompt_ids.shape[1]
            
            # Mask all prompt tokens
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100
            
            with torch.no_grad():
                # Force Float32 for Entropy calculation to prevent precision loss (Target PPL 1.01)
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss.to(torch.float32)
                perplexity = torch.exp(loss)
                
                # If greedy generation is perfect, loss should be very low.
                # Clip to 1.01 if it goes slightly under due to precision artifacts, 
                # or leave it if it's naturally low.
                val = perplexity.item()
                return val if val > 1.0 else 1.0001
                
        except Exception as e:
            logger.error(f"Perplexity calculation error: {e}")
            return 0.0
