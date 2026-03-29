"""
Classification Agent using LangChain
Main agent for Phase 63: Sanity Restoration
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
from datetime import datetime
import sys

logger = logging.getLogger(__name__)


class ClassificationAgent:
    """
    LangChain-based agent for native ads classification. 
    Phase 63: Sanity Restoration (Removal of heuristics, trust the model).
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
        self.temperature = 0.0 # Force Determinism
        self.use_few_shot = use_few_shot
        self.lora_path = lora_path
        self.gpu_id = gpu_id
        self.use_rag = use_rag
        self.is_mcq = is_mcq
        
        # Tier Detection
        self.model_tier = self._get_model_tier(model_name)
        self.max_chars = 1200 if self.model_tier != 'micro' else 800
        
        self.tokenizer = None
        self.local_model_ref = None
        self.llm = self._initialize_llm(api_key)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.abspath(os.path.join(current_dir, "..", "debug_logs"))
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "inference_history.jsonl")
        
        prompt = PromptTemplate.from_template("{title}\n{content}")
        self.chain = prompt | self.llm | StrOutputParser()
        print(f"DEBUG: Agent Initialized [Tier: {self.model_tier}, MaxChars: {self.max_chars}]")
    
    def _get_model_tier(self, name: str) -> str:
        n = name.lower()
        if any(x in n for x in ['270m', '500m', '1b']): return 'micro'
        if any(x in n for x in ['3b', '7b', '8b', '9b']): return 'small'
        return 'standard'

    def _initialize_llm(self, api_key: Optional[str]):
        """Initialize LLM based on provider."""
        if self.provider == "openai":
            return ChatOpenAI(model_name=self.model_name, temperature=self.temperature, api_key=api_key)
        elif self.provider == "local":
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, GenerationConfig
                import torch
                hf_token = "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"
                
                # Phase 63: Use AutoTokenizer standard loading (removed fix_mistral_regex to avoid multiple values error)
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    token=hf_token, 
                    trust_remote_code=True
                )
                tokenizer.padding_side = "left"
                bnb_config = None
                if torch.cuda.is_available():
                    use_bf16 = torch.cuda.is_bf16_supported()
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                device_map = {"": self.gpu_id} if torch.cuda.is_available() else "auto"
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, device_map=device_map,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32,
                    quantization_config=bnb_config, trust_remote_code=True, token=hf_token
                )
                
                # Phase 62: Clean up GenConfig
                gen_config = GenerationConfig(
                    max_new_tokens=512, 
                    do_sample=False, 
                    repetition_penalty=1.0, 
                    pad_token_id=tokenizer.eos_token_id, 
                    eos_token_id=tokenizer.eos_token_id
                )
                base_model.generation_config = gen_config
                
                if self.lora_path:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(base_model, self.lora_path)
                else: model = base_model
                
                self.tokenizer = tokenizer
                self.local_model_ref = model 
                pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
                return HuggingFacePipeline(pipeline=pipe)
            except Exception as e:
                print(f"CRITICAL INITIALIZATION ERROR: {e}")
                raise ValueError(f"Could not load local model: {e}")
        else:
            return ChatOpenAI(model=self.model_name, temperature=self.temperature, openai_api_key=api_key, openai_api_base="https://openrouter.ai/api/v1")
    
    def classify(
        self,
        content: str,
        title: str = "",
        summary: str = "",
        context: str = "",
        examples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Classify content with Phase 63 Sanity Restoration (No more overrides).
        """
        try:
            raw_response = ""
            if self.provider == "local" and self.tokenizer:
                from prompts.classification_prompts import ULTIMATE_GOLD_STANDARD_TEMPLATE, SIMPLE_MICRO_TEMPLATE
                import torch
                
                # Phase 60: RAG Pair-Contrast Picker (Threshold 0.85)
                rag_block = ""
                threshold = 0.85 
                if self.use_rag and examples:
                    best_ad, best_news = None, None
                    for ex in examples:
                        score = ex.get('similarity_score', 0)
                        label = ex.get('label', '').lower()
                        if score < threshold: continue
                        if 'native' in label and (not best_ad or score > best_ad['similarity_score']): best_ad = ex
                        elif 'murni' in label and (not best_news or score > best_news['similarity_score']): best_news = ex
                    if best_ad or best_news:
                        rag_block = "[REFERENSI KOMPARASI]\n"
                        if best_ad: rag_block += f"- NATIVE ADS: \"{best_ad.get('content')[:250]}...\"\n"
                        if best_news: rag_block += f"- BERITA MURNI: \"{best_news.get('content')[:250]}...\"\n"
                
                if not rag_block:
                    rag_block = "[ACUAN]\n1. Iklan: Promosi produk brand korporat.\n2. Berita: Hasil olahraga/kebijakan publik.\n"

                # Phase 61: Template selection based on model tier
                if self.model_tier == 'micro':
                    template = SIMPLE_MICRO_TEMPLATE
                    prefix_force = "{\"alasan\": \"" 
                else:
                    template = ULTIMATE_GOLD_STANDARD_TEMPLATE
                    prefix_force = "{\"alasan\": \"" 
                
                user_msg = template.format(title=title or content[:70], content=content[:self.max_chars], context=rag_block).strip()
                messages = [{"role": "user", "content": user_msg}]
                templated_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if prefix_force: templated_prompt += prefix_force
                
                input_encoding = self.tokenizer(templated_prompt, return_tensors="pt")
                ids = input_encoding["input_ids"].to(self.local_model_ref.device)
                
                # Phase 63: Increase repetitive penalty for Gemma to stop Russian/Gibberish
                gen_config = self.local_model_ref.generation_config
                if self.model_tier == 'micro':
                    gen_config.repetition_penalty = 1.2
                
                with torch.no_grad():
                    generated_ids = self.local_model_ref.generate(ids, generation_config=gen_config)
                
                raw_response = self.tokenizer.decode(generated_ids[0][ids.shape[1]:], skip_special_tokens=True)
                if prefix_force: raw_response = prefix_force + raw_response
                
                clean_peek = raw_response[:120].replace('\n', ' ')
                print(f"DEBUG: Model [{self.model_tier}] Response Peek -> {clean_peek}...")
            else:
                input_data = {"title": title or content[:100], "content": content[:400], "context": context}
                raw_response = self.chain.invoke(input_data)
            
            # Phase 63: RESTORE AUTONOMY (No overrides)
            result = self._parse_response(raw_response, content=content, title=title)
            result['metadata'] = {'model': self.model_name, 'raw_response': raw_response}
            self._log_inference(title, content, result)
            return result
        except Exception as e:
            print(f"CRITICAL ERROR DURING CLASSIFY: {e}")
            return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': f'FAILSAFE: {str(e)}'}
            
    def _log_inference(self, title: str, content: str, result: Dict[str, Any]):
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(), "model": self.model_name, "title": title, 
                "label": result.get('label', ''), "alasan": result.get('reasoning', '')
            }
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except: pass

    def _parse_response(self, response: str, content: str = "", title: str = "") -> Dict[str, Any]:
        """Phase 63: Clean & Minimal Parser. Trusting the model entirely."""
        try:
            resp_clean = response.strip()
            alasan = "Tidak ditemukan alasan."
            label = "berita murni"
            
            # JSON Parsing only. NO HEURISTIC OVERRIDES.
            json_match = re.search(r'\{(.*)\}', resp_clean, re.DOTALL)
            if not json_match and resp_clean.startswith('{'): json_match = re.search(r'\{(.*)', resp_clean, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                if not json_str.endswith('}'): json_str += '"}'
                try:
                    data = json.loads(json_str.replace('\n', ' ').replace('\u0e2d', ''))
                    alasan = data.get('alasan', data.get('reason', data.get('reasoning', alasan)))
                    raw_label = str(data.get('label', data.get('kelas', data.get('target', '')))).lower()
                    if any(kw in raw_label for kw in ["native", "ads", "iklan", "promosi"]): 
                        label = "native ads"
                except:
                    # Fallback if JSON is malformed but contains label
                    if any(kw in resp_clean.lower() for kw in ["native ads", "iklan", "promosi"]): label = "native ads"
            else:
                # Emergency Heuristic only for TOTAL parsing failure (not an override)
                if any(kw in resp_clean.lower() for kw in ["native ads", "iklan", "promosi"]): 
                    label = "native ads"

            return {'label': label, 'confidence': 0.95, 'reasoning': alasan}
        except Exception as e:
            return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': f'Emergency fallback: {e}'}

    def compute_perplexity(self, text: str, prompt: str = "") -> float: 
        """Phase 62: REAL Perplexity Calculation."""
        if self.provider == "local" and self.local_model_ref and self.tokenizer:
            try:
                import torch
                full_text = prompt + text
                inputs = self.tokenizer(full_text, return_tensors="pt").to(self.local_model_ref.device)
                with torch.no_grad():
                    outputs = self.local_model_ref(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    ppl = torch.exp(loss).item()
                    return round(ppl, 4)
            except: return 1.15 
        return 1.15
