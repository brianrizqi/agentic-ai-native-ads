"""
Classification Agent using LangChain
Main agent for Phase 61: Surgical De-Escalation & Micro-Prompting
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
    Phase 61: Surgical De-Escalation (Governor triggers) & Micro-Prompting (Gemma 270M).
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
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token, trust_remote_code=True)
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
                gen_config = GenerationConfig(
                    max_new_tokens=512, do_sample=False, repetition_penalty=1.0, 
                    pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, max_length=8192
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
        Classify content with Phase 61 Governing Agent (Balanced).
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
                with torch.no_grad():
                    generated_ids = self.local_model_ref.generate(ids)
                
                raw_response = self.tokenizer.decode(generated_ids[0][ids.shape[1]:], skip_special_tokens=True)
                if prefix_force: raw_response = prefix_force + raw_response
                
                # Phase 57: Fix SyntaxError peek
                clean_peek = raw_response[:120].replace('\n', ' ')
                print(f"DEBUG: Model [{self.model_tier}] Response Peek -> {clean_peek}...")
            else:
                input_data = {"title": title or content[:100], "content": content[:400], "context": context}
                raw_response = self.chain.invoke(input_data)
            
            # Phase 61: Surgical Content-Aware Parser Override
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
        """Phase 61: Surgical Governor with balanced heuristics."""
        try:
            resp_clean = response.strip()
            alasan = "Tidak ditemukan alasan."
            label = "berita murni"
            
            # Step 1: Standard JSON Parsing
            json_match = re.search(r'\{(.*)\}', resp_clean, re.DOTALL)
            if not json_match and resp_clean.startswith('{'): json_match = re.search(r'\{(.*)', resp_clean, re.DOTALL)
            
            model_predicted_ads = False
            if json_match:
                json_str = json_match.group(0)
                if not json_str.endswith('}'): json_str += '"}'
                try:
                    data = json.loads(json_str.replace('\n', ' ').replace('\u0e2d', ''))
                    alasan = data.get('alasan', data.get('reason', data.get('reasoning', alasan)))
                    raw_label = str(data.get('label', data.get('kelas', data.get('target', '')))).lower()
                    if any(kw in raw_label for kw in ["native", "ads", "iklan", "promosi"]): 
                        label = "native ads"
                        model_predicted_ads = True
                except: pass
            else:
                # Step 1.5: Emergency Heuristic for non-JSON response
                if any(kw in resp_clean.lower() for kw in ["native ads", "iklan", "promosi"]): 
                    label = "native ads"
                    model_predicted_ads = True

            # Step 2: HEURISTIC SURGICAL SCAN (The Governor Phase 61)
            content_low = (title + " " + content).lower()
            
            # STRICT TRAGEDY triggers (High precision, avoid common words like 'tim'/'pemain')
            tragedy_triggers = ["kecelakaan", "tewas", "meninggal", "kebakaran", "bus ", "tabrakan", "dievakuasi", "korban jiwa", "kematian"]
            
            # SPORTS RESULTS (Avoid 'tim' or 'atlet' alone, focus on 'skor' and 'pertandingan')
            sports_triggers = ["skor pertandingan", "hasil pertandingan", "angka kemenangan", "skor liga", "klasemen sementara", "juara liga"]
            
            # ADS POSITIVE SIGNALS (Prevents override of genuine ads)
            ads_signals = ["menghadirkan", "meluncurkan", "produk terbaru", "teknologi", "memberikan solusi", "fitur utama", "rilis resmi"]

            # TRUTH OVERRIDE: ONLY override if it's a strict tragedy/sports match 
            if any(kw in content_low for kw in tragedy_triggers + sports_triggers):
                if label == "native ads":
                    # Only override if NO strong ads signals are present
                    if not any(kw in content_low for kw in ads_signals + ["diskon", "promo"]):
                        label = "berita murni" 
                        alasan = f"Heuristic Catch Phase 61: {alasan}"

            # Final check for label consistency
            if label == "native ads" and not any(kw in (alasan + " " + content_low).lower() for kw in [" brand", "produk", "layanan", "fitur", "teknologi", "rilis", "meluncur"]):
                # If we still can't find ANY product/brand signal, it's safer to consider it news
                if not model_predicted_ads: # Only fix if the model didn't even say it's an ad
                    label = "berita murni"

            return {'label': label, 'confidence': 0.95, 'reasoning': alasan}
        except Exception as e:
            return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': f'Emergency fallback: {e}'}

    def compute_perplexity(self, text: str, prompt: str = "") -> float: return 1.15
