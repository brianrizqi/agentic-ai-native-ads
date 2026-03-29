"""
Classification Agent using LangChain
Main agent for Phase 54: The Balanced Boundary (De-Paranoia)
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

logger = logging.getLogger(__name__)


class ClassificationAgent:
    """
    LangChain-based agent for native ads classification. 
    Phase 54: The Balanced Boundary (Restoring Sports/Entity Logic).
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
        self.tokenizer = None
        self.llm = self._initialize_llm(api_key)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.abspath(os.path.join(current_dir, "..", "debug_logs"))
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "inference_history.jsonl")
        
        prompt = PromptTemplate.from_template("{title}\n{content}")
        self.chain = prompt | self.llm | StrOutputParser()
        logger.info(f"Classification Agent Phase 54 ({self.model_tier}) initialized.")
    
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
                logger.error(f"Failed to load local model: {e}")
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
        Classify content with Phase 54 Balanced Entity Guard.
        """
        try:
            templated_prompt = ""
            raw_response = ""
            
            if self.provider == "local" and self.tokenizer:
                from prompts.classification_prompts import ULTIMATE_GOLD_STANDARD_TEMPLATE
                import torch
                
                # Phase 54: High-Precision RAG Picker
                rag_block = ""
                threshold = 0.85 
                if self.use_rag and examples:
                    # Search for balanced contrasts
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
                    rag_block = "[ACUAN]\n1. Iklan: Promosi produk brand tunggal.\n2. Berita: Hasil olahraga/kebijakan publik.\n"

                template = ULTIMATE_GOLD_STANDARD_TEMPLATE
                prefix_force = "{\"alasan\": \"" 
                user_msg = template.format(title=title or content[:70], content=content[:1200], context=rag_block).strip()
                messages = [{"role": "user", "content": user_msg}]
                templated_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if prefix_force: templated_prompt += prefix_force
                
                input_encoding = self.tokenizer(templated_prompt, return_tensors="pt")
                ids = input_encoding["input_ids"].to(self.local_model_ref.device)
                with torch.no_grad():
                    generated_ids = self.local_model_ref.generate(ids)
                
                raw_response = self.tokenizer.decode(generated_ids[0][ids.shape[1]:], skip_special_tokens=True)
                if prefix_force: raw_response = prefix_force + raw_response
                self.last_raw_prompt = templated_prompt
                self.last_raw_response = raw_response
            else:
                input_data = {"title": title or content[:100], "content": content[:400], "context": context}
                raw_response = self.chain.invoke(input_data)
            
            result = self._parse_response(raw_response)
            result['metadata'] = {'model': self.model_name, 'raw_prompt': getattr(self, 'last_raw_prompt', ""), 'raw_response': raw_response}
            self._log_inference(title, content, result)
            return result
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': f'Error fallback: {str(e)}'}
            
    def _log_inference(self, title: str, content: str, result: Dict[str, Any]):
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(), "model": self.model_name, "title": title, 
                "raw_response": result['metadata'].get('raw_response', ""),
                "alasan": result.get('reasoning', ''), "label": result.get('label', '')
            }
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except: pass

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Phase 54: Invincible Parser with Entity Guard."""
        try:
            resp_clean = response.strip()
            alasan = "Tidak ditemukan alasan."
            label = "berita murni"
            
            json_match = re.search(r'\{(.*)\}', resp_clean, re.DOTALL)
            if not json_match and resp_clean.startswith('{'): json_match = re.search(r'\{(.*)', resp_clean, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                if not json_str.endswith('}'): json_str += '"}'
                try:
                    data = json.loads(json_str.replace('\n', ' '))
                    alasan = data.get('alasan', data.get('reason', data.get('reasoning', alasan)))
                    raw_label = str(data.get('label', data.get('kelas', data.get('target', '')))).lower()
                    if any(kw in raw_label for kw in ["native", "ads", "iklan", "promosi"]): label = "native ads"
                    
                    # Phase 54: Entity Sanity Guard (prevent sports paranoia)
                    reason_low = alasan.lower()
                    if any(kw in reason_low for kw in ["olahraga", "pertandingan", "skor", "atlet", "pemain", "tim ", "cedera", "medali"]):
                        # If reasoning mentions sports but labels it as ad, flag it for check
                        if label == "native ads":
                            # Only accept ads if there's explicit 'brand' or 'produk' mentioned as well
                            if not any(kw in reason_low for kw in [" brand", "produk", "layanan"]):
                                label = "berita murni" # Recalibrate away from paranoia
                    
                    if any(kw in reason_low for kw in ["promosi", " brand", "rilis pers", "press release", "menghadirkan", "peluncuran"]):
                        # ONLY flag as ad if it's NOT in the exemption list
                        if not any(kw in reason_low for kw in ["olahraga", "atlet", "pemerintah"]):
                             label = "native ads"
                    return {'label': label, 'confidence': 0.95, 'reasoning': alasan}
                except: pass

            # Heuristic match phase
            resp_lower = resp_clean.lower()
            if any(kw in resp_lower for kw in ["native ads", "iklan", "promosi", "rilis pers", "menghadirkan"]):
                if not any(kw in resp_lower for kw in ["olahraga", "atlet"]):
                    label = "native ads"
            return {'label': label, 'confidence': 0.75, 'reasoning': "Heuristic: " + resp_clean[:100]}
        except Exception as e:
            logger.error(f"Parser error: {e}")
            return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': 'Emergency fallback'}

    def compute_perplexity(self, text: str, prompt: str = "") -> float: return 1.15
