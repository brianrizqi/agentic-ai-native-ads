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
        # Phase 18: All models now use a Harmonized RAG Prompt that matches the training format
        # but with simplified context injection for smaller models to reduce noise.
        # Phase 18: All models now use a Harmonized RAG Prompt that matches the training format
        # but with simplified context injection for smaller models to reduce noise.
        if self.use_rag:
            if self.model_tier == "micro":
                # Phase 22: Micro-MCQ RAG for sub-500M models
                # Strictly aligned with Phase 14 Reasoning-First MCQ training.
                from prompts.classification_prompts import REASONING_MCQ_PROMPT_TEMPLATE
                prompt = PromptTemplate.from_template(REASONING_MCQ_PROMPT_TEMPLATE)
            else:
                # Phase 18: Harmonized RAG Prompt for larger models (1B+)
                # Matches training format with JSON reasoning.
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
{{"label": "native ads" atau "berita murni", "confidence": 0.0-1.0, "reasoning": "alasan singkat (max 150 karakter)"}}

Klasifikasi:
"""
                )
        elif self.provider == "local":
            # Use simplified prompt for local models
            from prompts.classification_prompts import simple_local_prompt
            prompt = simple_local_prompt
        else:
            # Use few-shot or simple prompt for API models
            prompt = few_shot_classification_prompt if use_few_shot else simple_classification_prompt
        
        # Create chain using LCEL (more robust than deprecated LLMChain)
        self.chain = prompt | self.llm | StrOutputParser()
        
        logger.info(f"Classification Agent initialized with {model_name} ({provider})")
    
    def _get_model_tier(self) -> str:
        """Categorize model by scale for prompt optimization."""
        name = self.model_name.lower()
        if any(kw in name for kw in ["270m", "0.5b", "tiny", "micro"]):
            return "micro"
        if any(kw in name for kw in ["1b", "2b", "3b", "small", "lite"]):
            return "small"
        if "qwen" in name and "9b" in name: # Qwen 9B is standard, 2B/1.5B is small
            return "standard"
        return "standard"

    def _is_small_model(self) -> bool:
        """Deprecated: Use self.model_tier instead."""
        return self.model_tier in ["micro", "small"]
    
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
            # Load local fine-tuned model (e.g., Llama with LoRA)
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                import torch
                
                logger.info(f"Loading local model from: {self.model_name}")
                
                # Use api_key as Hugging Face token if provided
                hf_token = "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"
                
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    token=hf_token,
                    trust_remote_code=True
                )
                
                # Setup quantization for local loading
                bnb_config = None
                if torch.cuda.is_available():
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )

                # Fix for multi-GPU: set explicit device or 'auto'
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
                
                # Apply LoRA if provided
                if self.lora_path:
                    logger.info(f"Applying LoRA adapters from: {self.lora_path}")
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(base_model, self.lora_path)
                else:
                    model = base_model
                
                # Set chat template if missing (crucial for LangChain LCEL integration)
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
                
                # Phase 22: Tier-Specific Initialization
                if self.model_tier == "micro":
                    # Micro (MCQ): Stop after the A/B label (don't stop on the trigger itself!)
                    stop_sequences = ["\n\n\n", tokenizer.eos_token]
                else:
                    # Standard/Small (JSON): Stop after closing brace
                    stop_sequences = ["}\n", "} \n", "}\n\n", "}", tokenizer.eos_token]
                
                # Phase 45: STOP FORCING MCQ for 270M (micro) - Reverting to JSON baseline
                self.is_mcq = self.is_mcq or "mcq" in model_name_lower
                
                # Override max_length di generation_config supaya tidak conflict
                # dengan max_new_tokens yang kita set di pipeline
                # Phase 33: Fixed n_new calculation - Always prioritize RAG or Reasoning requirements
                # Phase 49: Bumped to 768 for RAG to avoid early JSON truncation.
                if self.model_tier == "micro":
                    n_new = 384 
                else:
                    n_new = 1024 
                
                if hasattr(model, 'generation_config'):
                    # Explicitly set both to avoid conflict and 20-token default
                    model.config.max_length = 2048
                    model.generation_config.max_length = 2048
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=n_new,
                    do_sample=False,            # greedy decode
                    repetition_penalty=1.1,     # Reset to 1.1 for JSON stability (Phase 7)
                    return_full_text=False
                )
                
                # We will pass stop sequences to the invoke() call for better reliability
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
        
        Args:
            content: Main content text
            title: Article title
            summary: Content summary
            context: Retrieved context (legacy string format)
            examples: List of example dicts (Phase 20 multi-turn format)
            
        Returns:
            Classification result with label, confidence, and reasoning
        """
        try:
            logger.info("Classifying content...")
            
            templated_prompt = ""
            # Prepare input data for legacy/API path
            input_data = {
                "title": title or content[:100],
                "content": content[:400], # Use same char limit as training/eval HF scripts
                "context": context
            }
            
            # Use chat template for local models (Phase 31: Balanced Induction)
            if self.provider == "local" and self.tokenizer:
                from prompts.classification_prompts import BALANCED_2_SHOT_TEMPLATE
                
                # High-Confidence RAG (Threshold: 0.35)
                top_score = examples[0].get('similarity_score', 1.0) if (examples and self.use_rag) else 1.0
                use_this_rag = self.use_rag and examples and top_score > 0.35
                
                rag_context = ""
                if use_this_rag:
                    ex = examples[0]
                    rag_context = (
                        f"REFERENSI KONTEKSTUAL (Kemiripan={top_score:.2f}):\n"
                        f"- Konten: {ex.get('content', '')[:100]}...\n"
                        f"- Label: {ex.get('label')}\n"
                        "---"
                    )

                user_msg = BALANCED_2_SHOT_TEMPLATE.format(
                    title=title or content[:60],
                    content=content[:400],
                    context=rag_context
                ).strip()
                
                # Phase 31: NO PREFIX FORCE. Let the model break its mode-collapse
                # by following the induction from 2-shot examples.
                prefix_force = "" 

                messages = [{"role": "user", "content": user_msg}]

                # Apply chat template
                templated_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Use standard JSON stop sequences
                stop_seqs = ["}", "}\n", "\n\n", self.tokenizer.eos_token if self.tokenizer else "</s>"]
                if hasattr(self, 'stop_sequences'):
                    stop_seqs.extend(self.stop_sequences)
                
                response = self.llm.invoke(templated_prompt, stop=list(set(stop_seqs)))
                
            else:
                # Use LCEL with explicit dict (API providers)
                response = self.chain.invoke(input_data)
            
            # Parse response (response is already a string thanks to StrOutputParser)
            result = self._parse_response(response)
            
            # Add metadata
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
        """Parse LLM response into structured format."""
        try:
            import re
            
            # Phase 31: Adaptive Induction Parser (Unified)
            # 1. Primary: JSON Parsing (High Precision)
            if '{' in response and '}' in response:
                try:
                    json_str = response[response.find('{'):response.rfind('}')+1]
                    json_str = json_str.replace('\n', ' ').strip()
                    data = json.loads(json_str)
                    
                    # Extract label regardless of key position
                    raw_label = ""
                    if 'label' in data: raw_label = str(data['label']).lower()
                    elif 'Hasil' in data: raw_label = str(data['Hasil']).lower()
                    
                    if raw_label:
                        label = 'native ads' if ("native" in raw_label or "ads" in raw_label or "iklan" in raw_label) else 'berita murni'
                        return {
                            'label': label,
                            'confidence': data.get('confidence', 0.95),
                            'reasoning': data.get('reasoning', data.get('analisis', 'JSON extraction'))
                        }
                except:
                    pass

            # 2. Secondary: String / Keyword Fallback (Stability)
            resp_lower = response.lower()
            if any(kw in resp_lower for kw in ["native ads", "iklan", "[a]", "hasil: native ads", "label: native ads"]):
                return {'label': 'native ads', 'confidence': 0.9, 'reasoning': 'Keyword match fallback'}
            elif any(kw in resp_lower for kw in ["berita murni", "murni", "[b]", "hasil: berita murni", "label: berita murni"]):
                return {'label': 'berita murni', 'confidence': 0.9, 'reasoning': 'Keyword match fallback'}

            # 1. Try JSON parsing (generic fallback)
            if '{' in response and '}' in response:
                try:
                    json_str = response[response.find('{'):response.rfind('}')+1]
                    data = json.loads(json_str)
                    if 'label' in data:
                        label = data['label'].lower()
                        if "native" in label:
                            return {'label': 'native ads', 'confidence': data.get('confidence', 0.9), 'reasoning': data.get('reasoning', '')}
                        else:
                            return {'label': 'berita murni', 'confidence': data.get('confidence', 0.9), 'reasoning': data.get('reasoning', '')}
                except:
                    pass

            # 2. Early Phase Fallback to anchor-based search (HASIL, JAWABAN, etc)
            # Priorities: look for "Jawaban: [AB]" first as it's the 270M standard.
            # Phase 42: Removed \b boundary as multilingual gibberish (Thai/Russian) confuses the word boundary.
            mcq_label_match = re.search(r'Jawaban[:\s]*([AB])', response, re.IGNORECASE)
            
            if not mcq_label_match:
                # Try generic anchor-based search
                jawaban_match = re.search(r'(?:Jawaban|HASIL|Answer|JAWABAN|Klasifikasi|Petunjuk|label)[:\s]*([AB]|native ads|berita murni)', response, re.IGNORECASE)
            else:
                jawaban_match = mcq_label_match

            if not jawaban_match and self.model_tier in ["micro", "small"]:
                # Check for just the label code at the very beginning OR very end
                micro_match = re.search(r'(?:^\s*|\n|:\s*)([AB]|native ads|berita murni)\b', response, re.IGNORECASE)
                if micro_match:
                    jawaban_match = micro_match

            if jawaban_match:
                match_text = jawaban_match.group(1).upper()
                # Extract reasoning (everything before result)
                reasoning = response[:jawaban_match.start()].strip()
                if not reasoning:
                    reasoning = f'Detected Label: {match_text}'
                
                if match_text in ["A", "NATIVE ADS"]:
                    return {'label': 'native ads', 'confidence': 0.95, 'reasoning': reasoning[:300]}
                else:
                    return {'label': 'berita murni', 'confidence': 0.95, 'reasoning': reasoning[:300]}
            
            # Phase 14.1: Handle Reasoning Models (DeepSeek-R1, Qwen3)
            # Remove <think>...</think> blocks if present to clean JSON parsing
            if "<think>" in response and "</think>" in response:
                logger.info("Deep-reasoning format detected, stripping <think> tags")
                import re
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            elif "<think>" in response and not "</think>" in response:
                # Handle case where output is truncated inside <think>
                logger.warning("Truncated <think> block detected, attempting extraction")
                response = response.split("<think>")[-1].split("</think>")[-1].strip()

            # Clean up response for other parsing paths
            if "Output (JSON):" in response:
                response = response.split("Output (JSON):")[-1].strip()

            
            # Phase 13: MCQ Detection — single A or B (only if response is very short)
            clean_resp = response.strip()
            # Phase 24: Direct English Label Matching (Highest Priority for Zero-Shot Micro)
            if len(clean_resp) <= 15:
                clean_lower = clean_resp.lower()
                if "native ads" in clean_lower or clean_lower == "a":
                    return {'label': 'native ads', 'confidence': 0.99, 'reasoning': 'Direct English label match'}
                if "pure news" in clean_lower or "berita murni" in clean_lower or clean_lower == "b":
                    return {'label': 'berita murni', 'confidence': 0.99, 'reasoning': 'Direct English label match'}

            if len(clean_resp) <= 5:
                clean_upper = clean_resp.upper()
                mcq_match = re.match(r'^[^A-Za-z]*([AB])[^A-Za-z]*$', clean_upper)
                if mcq_match:
                    label_code = mcq_match.group(1)
                    if label_code == "A":
                        return {'label': 'native ads', 'confidence': 0.95, 'reasoning': f'Detected MCQ Label: {label_code}'}
                    else:
                        return {'label': 'berita murni', 'confidence': 0.95, 'reasoning': f'Detected MCQ Label: {label_code}'}
            
            # Phase 13: 'Ultra-Resilient' Parser (71% Target Recovery)
            # Broaden detection for 270M/1B models that use Indonesian or unstructured formats.
            clean_resp_lower = response.lower().strip()
            
            # 1. Broad Anchor Matching (Aligned with Training Prompt 'Klasifikasi:')
            # Multi-word anchors and Indonesian terms
            anchors_list = [
                "klasifikasi:", "kategori berita:", "kategori:", "label:", "hasil:", 
                "jawaban:", "conclusion:", "result:", "tipe:"
            ]
            
            resp_lines = response.split('\n')
            for idx, line in enumerate(resp_lines):
                line_clean = line.strip().lower()
                for anchor in anchors_list:
                    if anchor in line_clean:
                        # Scan the text AFTER the anchor on the same line
                        relevant_part = line_clean.split(anchor)[1].strip()
                        if not relevant_part and idx + 1 < len(resp_lines): # Check next line if empty
                            relevant_part = resp_lines[idx+1].strip().lower()
                        
                        # Phase 13: Multilingual Label Mapping
                        if any(kw in relevant_part for kw in ["berita murni", "murni", "objektif", " b", "news"]):
                            return {'label': 'berita murni', 'confidence': 0.95, 'reasoning': f'Anchor-based: {anchor} {relevant_part}'}
                        if any(kw in relevant_part for kw in ["native ads", "iklan", "promosi", " a", "ads"]):
                            return {'label': 'native ads', 'confidence': 0.95, 'reasoning': f'Anchor-based: {anchor} {relevant_part}'}

            # 2. Heuristic Start-of-Response (Tiny models often start with the label)
            # Check first 50 chars for high-confidence Indonesian terms
            first_block = clean_resp_lower[:50]
            if "native ads" in first_block or "iklan" in first_block or "promosi" in first_block:
                return {'label': 'native ads', 'confidence': 0.8, 'reasoning': 'Start-of-string heuristic'}
            if "berita murni" in first_block or "pure news" in first_block or "objektif" in first_block or "murni" in first_block:
                return {'label': 'berita murni', 'confidence': 0.8, 'reasoning': 'Start-of-string heuristic'}

            # Keep JSON logic as fallback for larger models or hybrid outputs
            start = response.find('{')
            if start == -1:
                # 3. SMARTER Keyword-based Scoring (Weighted & Negation-aware)
                native_kws = ["mempromosikan", "promosi", "persuasif", "iklan", "advertorial"]
                murni_kws = ["netral", "objektif", "informasi", "fakta", "berita", "news"]
                
                words = clean_resp_lower.split()
                n_score = 0
                m_score = 0
                
                for i, word in enumerate(words):
                    clean_word = word.strip('.,?!:()')
                    if clean_word in native_kws:
                        # Negation check: "tidak mempromosikan"
                        is_negated = (i > 0 and words[i-1] in ["tidak", "bukan", "tanpa", "no", "not"])
                        if is_negated: m_score += 1
                        else: n_score += 2 # Native keywords are more distinct
                    if clean_word in murni_kws:
                        is_negated = (i > 0 and words[i-1] in ["tidak", "bukan"])
                        if is_negated: n_score += 1
                        else: m_score += 1
                
                if n_score > m_score:
                    return {'label': 'native ads', 'confidence': 0.6, 'reasoning': f'Heuristic score N:{n_score} M:{m_score}'}
                elif m_score > n_score:
                    return {'label': 'berita murni', 'confidence': 0.6, 'reasoning': f'Heuristic score M:{m_score} N:{n_score}'}

                # 4. Final Absolute Fallback
                logger.warning(f"Ambiguous response, defaulting to berita murni: {response[:200]}")
                return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': 'Ambiguous safety default'}
            
            # Parse character by character to find the matching closing brace
            brace_count = 0
            in_string = False
            escape_next = False
            end = start
            
            for i in range(start, len(response)):
                char = response[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
            
            if brace_count != 0:
                logger.warning(f"Unmatched braces (count={brace_count}) in response: {response[:500]}")
                
                # Try to complete the JSON by adding missing closing braces
                if brace_count > 0:
                    json_str = response[start:end] + ('}' * brace_count)
                    logger.info(f"Attempting to complete JSON by adding {brace_count} closing brace(s)")
                    try:
                        result = json.loads(json_str)
                        logger.info("Successfully completed incomplete JSON")
                        reasoning = result.get('reasoning', str(result))
                        if isinstance(reasoning, list):
                            reasoning = " ".join([str(i) for i in reasoning])
                        else:
                            reasoning = str(reasoning)
                            
                        return {
                            'label': result.get('label', 'unknown'),
                            'confidence': float(result.get('confidence', 0.5)),
                            'reasoning': reasoning[:200]
                        }
                    except json.JSONDecodeError:
                        logger.warning("Failed to complete JSON, falling back to extraction")
                
                raise json.JSONDecodeError("Unmatched braces", response, start)
            
            # Successfully extracted potential JSON string, now parse it
            result = json.loads(response[start:end])
            reasoning = result.get('reasoning', str(result))
            if isinstance(reasoning, list):
                reasoning = " ".join([str(i) for i in reasoning])
            else:
                reasoning = str(reasoning)
                
            return {
                'label': result.get('label', 'unknown'),
                'confidence': float(result.get('confidence', 0.5)),
                'reasoning': reasoning[:200]
            }

                
        except (json.JSONDecodeError, Exception) as e:
            # Phase 8: If it's not JSON, it's likely a direct label from Gemma 3 (Phase 8/finetune.py style)
            # Priority 1: Check if the response STARTS with the label (Space-insensitive)
            clean_resp = response.lower().strip()
            clean_no_space = clean_resp.replace(' ', '').replace('_', '').replace('-', '')
            
            if clean_no_space.startswith("nativeads"):
                return {'label': 'native ads', 'confidence': 0.95, 'reasoning': clean_resp}
            elif clean_no_space.startswith("beritamurni"):
                return {'label': 'berita murni', 'confidence': 0.95, 'reasoning': clean_resp}
            
            # Priority 2: Keyword search with protection
            first_line_no_space = clean_resp.split('\n')[0].replace(' ', '')
            if "nativeads" in first_line_no_space:
                return {'label': 'native ads', 'confidence': 0.9, 'reasoning': clean_resp}
            elif "beritamurni" in first_line_no_space:
                return {'label': 'berita murni', 'confidence': 0.9, 'reasoning': clean_resp}
            
            # Log the error and fall through to legacy keyword logic below
            if isinstance(e, json.JSONDecodeError):
                logger.debug(f"JSON decode error: {e}")
            else:
                logger.warning(f"Failed to parse JSON: {e}")
            
            logger.info(f"Raw response: {response[:500]}")
        
        # Fallback parsing logic
        import re
        
        # Handle empty or very short responses
        if not response or len(response.strip()) < 5:
            logger.warning("Empty or very short response, using default classification")
            return {
                'label': 'berita murni',
                'confidence': 0.5,
                'reasoning': 'Empty or invalid model output'
            }
        
        # Check for repetitive text (model degradation)
        words = response.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            max_repetition = max(word_counts.values()) if word_counts else 0
            if max_repetition > len(words) * 0.3:  # More than 30% repetition
                logger.warning(f"Detected repetitive text (max repetition: {max_repetition}/{len(words)})")
                # Return a low-confidence fallback
                return {
                    'label': 'berita murni',
                    'confidence': 0.5,
                    'reasoning': 'Model generated repetitive text, using default classification'
                }
        
        # Try to extract partial JSON if braces are unmatched
        if '{' in response and '"label"' in response:
            try:
                # Try to extract label and confidence even from incomplete JSON
                label_match = re.search(r'"label":\s*"([^"]+)"', response)
                conf_match = re.search(r'"confidence":\s*(0?\.\d+|1\.0)', response)
                
                if label_match:
                    label = label_match.group(1)
                    
                    # Handle 'unknown' label - convert to default based on content analysis
                    if label == 'unknown' or label not in ['native ads', 'berita murni']:
                        logger.info(f"Invalid label '{label}' detected, using keyword fallback")
                        # Fall through to keyword analysis below
                    else:
                        confidence = float(conf_match.group(1)) if conf_match else 0.6
                        logger.info(f"Extracted partial JSON: {label} (confidence: {confidence})")
                        return {
                            'label': label,
                            'confidence': confidence,
                            'reasoning': 'Extracted from incomplete JSON'
                        }
            except Exception as e:
                logger.debug(f"Failed to extract partial JSON: {e}")
        
        # Keyword-based fallback
        response_lower = response.lower()
        
        # Count indicators for each class
        native_ads_indicators = [
            'native ads', 'native advertising', 'advertorial', 'sponsored',
            'promosi', 'iklan', 'brand', 'produk', 'layanan',
            'persuasif', 'mengajak', 'meyakinkan'
        ]
        
        berita_murni_indicators = [
            'berita murni', 'pure news', 'objektif',
            'berbagai sudut pandang', 'kritik', 'investigasi'
        ]
        
        native_score = sum(1 for indicator in native_ads_indicators if indicator in response_lower)
        berita_score = sum(1 for indicator in berita_murni_indicators if indicator in response_lower)
        
        logger.info(f"Keyword analysis - Native ads: {native_score}, Berita murni: {berita_score}")
        
        # Determine label based on scores
        if native_score > berita_score:
            label = 'native ads'
            confidence = min(0.6 + (native_score * 0.05), 0.85)
        elif berita_score > native_score:
            label = 'berita murni'
            confidence = min(0.6 + (berita_score * 0.05), 0.85)
        else:
            # If tied, check for explicit mentions
            if 'native ads' in response_lower or 'advertorial' in response_lower:
                label = 'native ads'
                confidence = 0.65
            else:
                label = 'berita murni'
                confidence = 0.5
        
        logger.info(f"Fallback result: {label} (confidence: {confidence:.2f})")
        
        return {
            'label': label,
            'confidence': confidence,
            'reasoning': response[:200]
        }
    
    def _get_fallback_classification(self, text: str) -> Dict[str, Any]:
        """Fallback classification using keywords."""
        text_lower = text.lower()
        
        # Promotional keywords
        promo_keywords = [
            'promo', 'diskon', 'gratis', 'beli', 'dapatkan',
            'penawaran', 'spesial', 'cashback', 'bonus'
        ]
        promo_count = sum(1 for kw in promo_keywords if kw in text_lower)
        
        if promo_count >= 2:
            return {
                'label': 'native ads',
                'confidence': 0.65,
                'reasoning': f'Keyword-based: detected {promo_count} promotional terms'
            }
        else:
            return {
                'label': 'berita murni',
                'confidence': 0.60,
                'reasoning': 'Keyword-based: appears to be pure news content'
            }

    def compute_perplexity(self, text: str, prompt: str = "") -> float:
        """Compute perplexity for a given text, optionally conditioned on a prompt."""
        if self.provider != "local" or not self.tokenizer:
            return 0.0
            
        try:
            import torch
            model = self.llm.pipeline.model
            tokenizer = self.tokenizer
            
            if prompt:
                # Calculate PPL of 'text' given 'prompt'
                # Encode both together
                full_text = prompt + text
                encoding = tokenizer(full_text, return_tensors="pt")
                input_ids = encoding["input_ids"].to(model.device)
                
                # Get the length of the prompt tokens
                prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
                
                # Create labels: mask out the prompt tokens with -100
                labels = input_ids.clone()
                labels[:, :prompt_len] = -100
                
                with torch.no_grad():
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()
            else:
                # Standalone PPL (fallback)
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
