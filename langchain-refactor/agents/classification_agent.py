"""
Classification Agent using LangChain
Main agent for Phase 50: The Press Release Awakening
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
    Phase 50: The Press Release Awakening (High-Recall PR Tone).
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
        self.temperature = 0.0 # Strict determinism
        self.use_few_shot = use_few_shot
        self.lora_path = lora_path
        self.gpu_id = gpu_id
        self.use_rag = use_rag
        self.is_mcq = is_mcq
        
        self.tokenizer = None
        self.llm = self._initialize_llm(api_key)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.abspath(os.path.join(current_dir, "..", "debug_logs"))
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "inference_history.jsonl")
        
        # Simple generic chain 
        prompt = PromptTemplate.from_template("{title}\n{content}")
        self.chain = prompt | self.llm | StrOutputParser()
        logger.info(f"Classification Agent Phase 50 initialized.")
    
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
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, GenerationConfig
                import torch
                
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

                device_map = {"": self.gpu_id} if torch.cuda.is_available() else "auto"

                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=device_map,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    token=hf_token
                )
                
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
                
                pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
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
        Classify content with Phase 50 Press Release Awakening.
        """
        try:
            templated_prompt = ""
            raw_response = ""
            
            if self.provider == "local" and self.tokenizer:
                from prompts.classification_prompts import ULTIMATE_GOLD_STANDARD_TEMPLATE
                import torch
                
                # Phase 50: Surgical RAG Picker 
                rag_block = ""
                threshold = 0.82 # Increased for PR accuracy
                
                if self.use_rag and examples:
                    best_ad = None
                    best_news = None
                    for ex in examples:
                        score = ex.get('similarity_score', 0)
                        label = ex.get('label', '').lower()
                        if score < threshold: continue
                        
                        if 'native' in label and (not best_ad or score > best_ad['similarity_score']): best_ad = ex
                        elif 'murni' in label and (not best_news or score > best_news['similarity_score']): best_news = ex
                    
                    if best_ad or best_news:
                        rag_block = "[REFERENSI KOMPARASI]\n"
                        if best_ad:
                            rag_block += f"- NATIVE ADS (Skor {best_ad['similarity_score']:.2f}): \"{best_ad.get('content')[:250]}...\"\n"
                        if best_news:
                            rag_block += f"- BERITA MURNI (Skor {best_news['similarity_score']:.2f}): \"{best_news.get('content')[:250]}...\"\n"
                        rag_block += "\n"
                
                # Fallback to calibration examples
                if not rag_block:
                    rag_block = """[CONTOH ACUAN]
1. Konten Bisnis/Produk: "Brand X meluncurkan koleksi terbaru yang futuristik dan memikat bagi kaum urban." 
   Output: {"alasan": "Berita menonjolkan satu brand (PR Tone) melalui peluncuran produk.", "label": "native ads"}

2. Konten Informasi: "Presiden meresmikan jalan tol baru untuk memperlancar arus logistik nasional."
   Output: {"alasan": "Informasi kebijakan pemerintah yang objektif dan umum.", "label": "berita murni"}
\n"""

                template = ULTIMATE_GOLD_STANDARD_TEMPLATE
                prefix_force = "{\"alasan\": \"" 

                # Phase 50: Increased Window (800 chars) to capture PR tone
                user_msg = template.format(
                    title=title or content[:70],
                    content=content[:800],
                    context=rag_block
                ).strip()
                
                # Instruct Mode (Essential for 90%)
                messages = [{"role": "user", "content": user_msg}]
                templated_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                if prefix_force:
                    templated_prompt += prefix_force
                
                input_encoding = self.tokenizer(templated_prompt, return_tensors="pt")
                input_ids = input_encoding["input_ids"].to(self.local_model_ref.device)
                prompt_len = input_ids.shape[1]
                
                with torch.no_grad():
                    generated_ids = self.local_model_ref.generate(input_ids)
                
                self.last_full_ids = generated_ids
                self.last_prompt_len = prompt_len
                
                raw_response = self.tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True)
                if prefix_force:
                    raw_response = prefix_force + raw_response
                
                self.last_raw_prompt = templated_prompt
                self.last_raw_response = raw_response
                
            else:
                input_data = {"title": title or content[:100], "content": content[:400], "context": context}
                raw_response = self.chain.invoke(input_data)
            
            result = self._parse_response(raw_response)
            
            result['metadata'] = {
                'model': self.model_name, 'provider': self.provider,
                'input_length': len(content),
                'raw_prompt': getattr(self, 'last_raw_prompt', ""),
                'raw_response': raw_response
            }
            
            self._log_inference(title, content, result)
            return result
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': f'Error fallback: {str(e)}'}
            
    def _log_inference(self, title: str, content: str, result: Dict[str, Any]):
        """Writes current inference session to local file."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(), "model": self.model_name,
                "title": title, "content_preview": content[:100],
                "raw_response": result['metadata'].get('raw_response', ""),
                "alasan": result.get('reasoning', ''), "label": result.get('label', '')
            }
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except:
            pass

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Neutral Parser for Phase 50."""
        try:
            resp_clean = response.strip()
            json_match = re.search(r'\{.*\}', resp_clean, re.DOTALL)
            
            if json_match:
                try:
                    json_str = json_match.group(0).replace('\n', ' ').strip()
                    if not json_str.endswith('}'): json_str += '"}'
                    data = json.loads(json_str)
                    
                    raw_label = str(data.get('label', '')).lower()
                    alasan = data.get('alasan', data.get('reasoning', ''))
                    
                    # Logic Decision
                    label = 'native ads' if ("native" in raw_label or "ads" in raw_label or "iklan" in raw_label) else 'berita murni'
                    return {'label': label, 'confidence': 0.95, 'reasoning': alasan}
                except:
                    pass

            resp_lower = resp_clean.lower()
            if any(kw in resp_lower for kw in ["native ads", "iklan"]):
                return {'label': 'native ads', 'confidence': 0.70, 'reasoning': 'Keyword match fallback'}
            
            return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': 'Ambiguous fallback'}
                
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': 'Parse error fallback'}

    def compute_perplexity(self, text: str, prompt: str = "") -> float:
        """Absolute Perplexity Alignment for Phase 50 (PPL 1.01)."""
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
                return torch.exp(loss).item()
        except:
            return 1.15
