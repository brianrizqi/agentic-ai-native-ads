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

from prompts.classification_prompts import few_shot_classification_prompt, simple_classification_prompt

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
        self.temperature = temperature
        self.use_few_shot = use_few_shot
        self.lora_path = lora_path
        self.gpu_id = gpu_id
        self.use_rag = use_rag
        self.is_mcq = is_mcq or use_rag  # RAG optimization defaults to MCQ
        
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
                
                # Add stop sequence for JSON block and to prevent babbling
                # Phase 22: Micro tier uses " " (space) or "\n" to force one-token MCQ
                if self.model_tier == "micro":
                    # NOTE: Do NOT include " " (space) — it cuts output before label is generated
                    stop_sequences = ["\n\n", tokenizer.eos_token]
                else:
                    stop_sequences = ["}\n", "} ", "\n\n", "Analisis:", "Artikel:", tokenizer.eos_token]
                
                # Use both name-based check and explicit flag
                is_mcq = self.is_mcq or ("gemma" in model_name_lower and "270" in model_name_lower) or "mcq" in model_name_lower
                
                # Override max_length di generation_config supaya tidak conflict
                # dengan max_new_tokens yang kita set di pipeline
                # Phase 33: Fixed n_new calculation - Always prioritize RAG or Reasoning requirements
                if self.use_rag:
                    n_new = 512
                else:
                    is_reasoning = "deepseek-r1" in model_name_lower or "qwen3-" in model_name_lower
                    n_new = 128 if is_mcq else (1024 if is_reasoning else 512)
                
                # Model-specific overrides
                if self.model_tier == "micro":
                    n_new = max(n_new, 256)
                
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
                    repetition_penalty=1.1,     # increase for stability
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
            
            # Use chat template for local models to ensure instruction following
            if self.provider == "local" and self.tokenizer:
                if self.use_rag and examples:
                    if self.model_tier in ["micro", "small"]:
                        # Identity Mirror: always use Training Format to avoid Instruction Fade.
                        # For micro, inject RAG only when similarity >= 0.75 (was 0.95, too strict).
                        top_similarity = examples[0].get('similarity_score', 0.0) if examples else 0.0
                        
                        # Phase 30: Use Reasoning-First MCQ for 270M to match training
                        from prompts.classification_prompts import REASONING_MCQ_PROMPT_TEMPLATE, TRAINING_PROMPT_TEMPLATE
                        
                        # Threshold handles noise vs signal (Phase 34: Distance-based, Lower is Better)
                        # For FAISS L2 distance: 0.0 is perfect, >1.3 is weak.
                        threshold = 1.1 if self.model_tier == "micro" else 1.25
                        
                        target_template = REASONING_MCQ_PROMPT_TEMPLATE if self.model_tier == "micro" else TRAINING_PROMPT_TEMPLATE
                        
                        if top_similarity <= threshold and examples:
                            # Split example into separate turn
                            ex = examples[0]
                            ex_content = ex.get('content', '')[:120].replace('\n', ' ')
                            ex_label = ex.get('label', 'unknown')
                            ex_reasoning = ex.get('reasoning', 'No reasoning available')[:100]
                            
                            # Ensure label matches A/B if using MCQ
                            if self.model_tier == "micro":
                                ex_label_short = "A" if "native" in ex_label.lower() else "B"
                                assistant_content = f"Analisis: {ex_reasoning}\nJawaban: {ex_label_short}"
                            else:
                                assistant_content = json.dumps({"label": ex_label, "confidence": 1.0, "reasoning": ex_reasoning})
                            
                            # Build 2rd turn messages (Phase 32: Multi-turn Shot-Injection)
                            # Phase 33: RESTORED Instructions and Analisis Trigger for Turn 2
                            print(f"DEBUG: RAG Triggered (Dist: {top_similarity:.4f})")
                            messages = [
                                {
                                    "role": "user", 
                                    "content": target_template.format(
                                        title=ex.get('title', 'Artikel Contoh'),
                                        content=ex_content,
                                        context=""
                                    ).split('Analisis:')[0].strip() + "\n\nAnalisis:"
                                },
                                {
                                    "role": "assistant",
                                    "content": assistant_content
                                },
                                {
                                    "role": "user",
                                    "content": f"TUGAS: Klasifikasikan sebagai 'native ads' atau 'berita murni' (Pilih A/B).\n\nJudul: {title or content[:60]}\nKonten: {content[:400]}\n\nAnalisis:"
                                }
                            ]
                        else:
                            # Pure Zero-Shot: verbatim training template (matches original baseline)
                            print(f"DEBUG: Using Zero-Shot (Dist: {top_similarity:.4f})")
                            user_msg = target_template.format(
                                title=title or content[:60],
                                content=content[:400],
                                context=""
                            )
                            # Strip literal placeholders left from formatting
                            user_msg = user_msg.replace('{context}', '').strip()
                            # Clean up trailing prompt anchors
                            if self.model_tier == "micro":
                                user_msg = user_msg.split('Analisis:')[0].strip() + "\n\nAnalisis:"
                            else:
                                user_msg = user_msg.split('Output (JSON):')[0].strip() + "\n\nOutput (JSON):"
                            
                            messages = [{"role": "user", "content": user_msg}]
                    else:
                        # Phase 20: Keep Multi-turn RAG for standard tier (8B+)
                        # Phase 20: Keep Multi-turn RAG for standard tier (8B+)
                        messages = []
                        full_template = self.chain.first.template
                        instructions = full_template.split('{context}')[0].split('Judul:')[0].strip()
                        
                        for i, ex in enumerate(examples):
                            ex_content = ex.get('content', '')[:150].replace('\n', ' ')
                            ex_label = ex.get('label', 'unknown')
                            ex_reasoning = ex.get('reasoning', 'No reasoning available')
                            
                            if i == 0:
                                user_msg = f"{instructions}\n\nCONTOH 1:\nKonten: {ex_content}"
                            else:
                                user_msg = f"CONTOH {i+1}:\nKonten: {ex_content}"
                            
                            messages.append({"role": "user", "content": user_msg})
                            messages.append({"role": "assistant", "content": json.dumps({
                                "label": ex_label, "confidence": 1.0, "reasoning": ex_reasoning[:150]
                            })})
                        
                        target_msg = f"TARGET UNTUK DIKLASIFIKASI:\nJudul: {title or content[:100]}\nKonten: {content[:400]}"
                        messages.append({"role": "user", "content": target_msg})
                else:
                    # standard monolithic path
                    messages = [{"role": "user", "content": self.chain.first.format(**input_data)}]
                
                # Use Chat Template only if it hasn't been bypassed by Raw Mode
                if not templated_prompt:
                    templated_prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                
                # Phase 23: Explicitly pass ALL stop sequences to invoke
                stop_seqs = getattr(self, 'stop_sequences', None)
                response = self.llm.invoke(templated_prompt, stop=stop_seqs)
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
            # Phase 28: Improved Robust Parsing for local models
            import re
            
            # 1. Try JSON parsing first for the "Training Case"
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
            mcq_label_match = re.search(r'Jawaban[:\s]*([AB])\b', response, re.IGNORECASE)
            
            if not mcq_label_match:
                # Try generic anchor-based search
                jawaban_match = re.search(r'(?:Jawaban|HASIL|Answer|JAWABAN|Klasifikasi|Petunjuk|label)[:\s]*([AB]|native ads|berita murni)\b', response, re.IGNORECASE)
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
            if len(clean_resp) <= 5:
                clean_upper = clean_resp.upper()
                mcq_match = re.match(r'^[^A-Za-z]*([AB])[^A-Za-z]*$', clean_upper)
                if mcq_match:
                    label_code = mcq_match.group(1)
                    if label_code == "A":
                        return {'label': 'native ads', 'confidence': 0.95, 'reasoning': f'Detected MCQ Label: {label_code}'}
                    else:
                        return {'label': 'berita murni', 'confidence': 0.95, 'reasoning': f'Detected MCQ Label: {label_code}'}
            
            # Phase 12: Fallback for raw labels (Space-insensitive)
            clean_resp_lower = response.lower().strip()
            clean_no_space = clean_resp_lower.replace(' ', '').replace('_', '').replace('-', '')
            
            if "nativeads" in clean_no_space[:100]:
                return {'label': 'native ads', 'confidence': 0.9, 'reasoning': 'Detected via label match (lenient).'}
            elif "beritamurni" in clean_no_space[:100]:
                return {'label': 'berita murni', 'confidence': 0.9, 'reasoning': 'Detected via label match (lenient).'}
            
            # Keep JSON logic as fallback for larger models or hybrid outputs
            start = response.find('{')
            if start == -1:
                # If no JSON and no startswith, try keyword search
                if "native ads" in clean_resp_lower[:100]:
                    return {'label': 'native ads', 'confidence': 0.9, 'reasoning': response[:200]}
                elif "berita murni" in clean_resp_lower[:100]:
                    return {'label': 'berita murni', 'confidence': 0.9, 'reasoning': response[:200]}
                
                # --- Extreme Fallback (Fuzzy Matching for 270M Model) ---
                native_ads_keywords = [
                    "mempromosikan", "promosi", "persuasif", "menarik", "positif", 
                    "memanjakan", "marketing", "copywriting"
                ]
                berita_murni_keywords = [
                    "netral", "objektif", "tanpa promosi", "kinerja/risiko", 
                    "menginformasikan"
                ]
                
                native_score = sum(1 for kw in native_ads_keywords if kw in clean_resp_lower)
                murni_score = sum(1 for kw in berita_murni_keywords if kw in clean_resp_lower)
                
                if native_score > murni_score:
                    return {'label': 'native ads', 'confidence': 0.5, 'reasoning': response[:200]}
                elif murni_score > native_score:
                    return {'label': 'berita murni', 'confidence': 0.5, 'reasoning': response[:200]}
                
                logger.warning(f"No label or JSON found in response: {response[:500]}")
                raise ValueError("No JSON or label found")
            
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
