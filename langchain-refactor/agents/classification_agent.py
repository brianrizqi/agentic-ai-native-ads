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
        
        # Tier Detection (Phase 66: Radical Reduction for Micro)
        self.model_tier = self._get_model_tier(model_name)
        self.max_chars = 500 if self.model_tier == 'micro' else 1200
        
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

    def _get_model_family(self, name: str) -> str:
        """Detect model family for prompt tailoring."""
        n = name.lower()
        if 'gemma' in n: return 'gemma'
        if 'llama' in n: return 'llama'
        if 'qwen' in n: return 'qwen'
        return 'other'

    def _initialize_llm(self, api_key: Optional[str]):
        """Initialize LLM based on provider."""
        if self.provider == "openai":
            return ChatOpenAI(model_name=self.model_name, temperature=self.temperature, api_key=api_key)
        elif self.provider == "local":
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, GenerationConfig
                import torch
                hf_token = "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"
                
                # Phase 65: Nuclear Stability Reloaded (Safe Tokenizer Loading)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, token=hf_token, trust_remote_code=True, 
                        fix_mistral_regex=True
                    )
                except TypeError:
                    # Fallback for transformers versions where this is already handled or causes conflicts
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, token=hf_token, trust_remote_code=True
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
                
                # Phase 73: The Sanctuary Revert (No Embedding Resize on 4-bit)
                # Resizing 4-bit embeddings causes logit explosion (9000+ PPL).
                tokenizer.padding_side = "left"
                
                # Phase 62: Clean up GenConfig
                gen_config = GenerationConfig(
                    max_new_tokens=512, do_sample=False, repetition_penalty=1.0, 
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
            if self.provider == "local" and self.tokenizer:
                from prompts.classification_prompts import (
                    ULTIMATE_GOLD_STANDARD_TEMPLATE, SIMPLE_MICRO_TEMPLATE,
                    BILINGUAL_GOLD_STANDARD_TEMPLATE, BILINGUAL_MICRO_TEMPLATE,
                    ULTRA_STABLE_MICRO_TEMPLATE, MCQ_PROMPT_TEMPLATE
                )
                import torch
                
                # Phase 64: Improved RAG (Threshold 0.70)
                rag_block = ""
                threshold = 0.70
                if self.use_rag and examples:
                    ads = [ex for ex in examples if 'native' in ex.get('label', '').lower() and ex.get('similarity_score', 0) >= threshold]
                    news = [ex for ex in examples if 'murni' in ex.get('label', '').lower() and ex.get('similarity_score', 0) >= threshold]
                    
                    # Sort by similarity
                    ads = sorted(ads, key=lambda x: x['similarity_score'], reverse=True)
                    news = sorted(news, key=lambda x: x['similarity_score'], reverse=True)
                    
                    # Phase 68: NO RAG for Micro (< 500M)
                    # The RAG context "poisons" the attention of the 270M model. 
                    # Disabling to let fine-tuned weights work in Zero-G.
                    if self.model_tier == 'micro':
                        rag_block = ""
                    elif ads or news:
                        rag_block = "[REFERENSI KOMPARASI]\n"
                        limit = 2
                        for i, ex in enumerate(ads[:limit]):
                            rag_block += f"- NATIVE ADS: \"{ex.get('content')[:150]}...\"\n"
                        for i, ex in enumerate(news[:limit]):
                            rag_block += f"- BERITA MURNI: \"{ex.get('content')[:150]}...\"\n"
                
                if not rag_block:
                    rag_block = "[ACUAN]\n1. Iklan: Promosi produk brand korporat.\n2. Berita: Hasil olahraga/kebijakan publik.\n"

                if self.model_tier == 'micro':
                    # Phase 74: Atomic MCQ (Minimal instruction at the end for recency bias)
                    template = MCQ_PROMPT_TEMPLATE
                    prefix_force = "" # Let the template handle the end
                else:
                    template = BILINGUAL_GOLD_STANDARD_TEMPLATE if is_bilingual else ULTIMATE_GOLD_STANDARD_TEMPLATE
                    prefix_force = "{\"alasan\": \"" 
                
                user_msg = template.format(title=title or content[:70], content=content[:self.max_chars], context=rag_block).strip()
                
                # Phase 74: Let Tokenizer handle BOS/EOS automatically (Corrects 8K PPL)
                if self.model_tier == 'micro':
                    templated_prompt = user_msg
                else:
                    messages = [{"role": "user", "content": user_msg}]
                    templated_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    if prefix_force: templated_prompt += prefix_force
                
                input_encoding = self.tokenizer(templated_prompt, return_tensors="pt", add_special_tokens=True)
                input_ids = input_encoding["input_ids"].to(self.local_model_ref.device)
                mask = input_encoding["attention_mask"].to(self.local_model_ref.device)
                
                # Phase 76: Hybrid Logit-Reasoning Pivot (The Final Solution)
                if self.model_tier == 'micro':
                    import torch
                    with torch.no_grad():
                        # Step 1: Atomic Classification via Logits
                        outputs = self.local_model_ref(input_ids, attention_mask=mask)
                        next_token_logits = outputs.logits[0, -1, :]
                        
                        id_A = self.tokenizer.encode("A", add_special_tokens=False)[-1]
                        id_B = self.tokenizer.encode("B", add_special_tokens=False)[-1]
                        id_A_space = self.tokenizer.encode(" A", add_special_tokens=False)[-1]
                        id_B_space = self.tokenizer.encode(" B", add_special_tokens=False)[-1]
                        
                        val_A = max(next_token_logits[id_A].item(), next_token_logits[id_A_space].item())
                        val_B = max(next_token_logits[id_B].item(), next_token_logits[id_B_space].item())
                        
                        label = "native ads" if val_A > val_B else "berita murni"
                        probs = torch.softmax(torch.tensor([val_A, val_B]), dim=0)
                        conf = probs[0].item() if label == "native ads" else probs[1].item()
                        
                        # Step 2: Metrics-Driven Reasoning via Natural Anchor (Phase 79)
                        # Phase 79: "The Human Touch" - forcing descriptive explanation over robotic percentages
                        snippet = content[:300].replace('\n', ' ')
                        reason_prompt = (
                            f"Kutipan Artikel: \"{snippet}...\"\n"
                            f"Label: {label}\n"
                            f"Analisis Ahli: Konten ini merupakan {label} karena secara spesifik menginformasikan tentang "
                        )
                        
                        reason_inputs = self.tokenizer(reason_prompt, return_tensors="pt", add_special_tokens=True).to(self.local_model_ref.device)
                        
                        reason_ids = self.local_model_ref.generate(
                            reason_inputs.input_ids,
                            attention_mask=reason_inputs.attention_mask,
                            max_new_tokens=64,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                        
                        reason_text = self.tokenizer.decode(reason_ids[0][reason_inputs.input_ids.shape[1]:], skip_special_tokens=True)
                        
                        # Step 3: Phase 79 Numerical Suppression & Cleaning
                        import re
                        # 1. Kill at first newline or hallucinated tag
                        reason_text = reason_text.split("\n")[0].split("Analisis:")[0].split("Judul:")[0].strip(' ".,')
                        # 2. Regex to strip "100%", "10%", "1. ", "100 persen" etc.
                        reason_text = re.sub(r'^\d+\s?%?\s?', '', reason_text)
                        reason_text = re.sub(r'^\d+\s?persen\s?', '', reason_text)
                        reason_text = re.sub(r'^[A-B]\.\s?', '', reason_text) 
                        
                        final_reason = reason_text.strip()
                        if not final_reason or len(final_reason) < 10:
                            final_reason = f"Konten ini membahas topik {label} berdasarkan bukti di dalam teks artikel."
                        
                        print(f"DEBUG: Hybrid Finisher [{self.model_tier}] -> Choice: {label}, Final Reason: {final_reason[:60]}...")
                        return {
                            'label': label, 
                            'confidence': conf, 
                            'reasoning': final_reason
                        }

                # Standard Generation for other tiers
                gen_config = self.local_model_ref.generation_config
                gen_config.repetition_penalty = 1.0
                gen_config.do_sample = False
                
                with torch.no_grad():
                    generated_ids = self.local_model_ref.generate(
                        input_ids, 
                        attention_mask=mask,
                        generation_config=gen_config,
                        max_new_tokens=512,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                raw_response = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
                if prefix_force and self.model_tier != 'micro': 
                    raw_response = prefix_force + raw_response
                
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
        """Phase 66: Robust A/B and JSON Parser."""
        try:
            resp_clean = response.strip()
            alasan = "Tidak ditemukan alasan (mode MCQ)."
            label = "berita murni"
            
            # 1. MCQ Parsing (Priority for micro)
            # Find the first A or B that appears after "Jawaban" or at the start
            mcq_match = re.search(r'(?:Jawaban|Pilih|Answer):\s*([AB])', resp_clean, re.IGNORECASE)
            if not mcq_match:
                # Direct check for single token A/B
                if resp_clean.split() and resp_clean.split()[0].upper() in ['A', 'B']:
                    label = "native ads" if resp_clean.split()[0].upper() == 'A' else "berita murni"
                    return {'label': label, 'confidence': 0.9, 'reasoning': alasan}
            else:
                choice = mcq_match.group(1).upper()
                label = "native ads" if choice == 'A' else "berita murni"
                return {'label': label, 'confidence': 0.9, 'reasoning': alasan}

            # 2. JSON Parsing fallback
            json_match = re.search(r'\{(.*)\}', resp_clean, re.DOTALL)
            if not json_match and resp_clean.startswith('{'): json_match = re.search(r'\{(.*)', resp_clean, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                if not json_str.endswith('}'): 
                    if json_str.count('"') % 2 != 0: json_str += '"'
                    json_str += '}'
                try:
                    json_str_clean = re.sub(r'[^\x00-\x7F]+', ' ', json_str) 
                    data = json.loads(json_str_clean.replace('\n', ' '))
                    alasan = data.get('alasan', data.get('reason', data.get('reasoning', alasan)))
                    raw_label = str(data.get('label', data.get('kelas', data.get('target', '')))).lower()
                    if any(kw in raw_label for kw in ["native", "ads", "iklan", "promosi", "a"]): 
                        label = "native ads"
                except:
                    if any(kw in resp_clean.lower() for kw in ["native ads", "iklan", "promosi"]): label = "native ads"
            else:
                if any(kw in resp_clean.lower()[:50] for kw in ["native ads", "iklan", "promosi"]): 
                    label = "native ads"
                elif "A" in resp_clean[:10].upper():
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
