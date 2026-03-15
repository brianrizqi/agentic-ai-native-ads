"""
Classification Agent using LangChain
Main agent for classifying content as native ads or pure news
"""

from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.output_parsers import StrOutputParser
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
        lora_path: Optional[str] = None
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
        """
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.use_few_shot = use_few_shot
        self.lora_path = lora_path
        
        # Initialize LLM
        self.tokenizer = None
        self.llm = self._initialize_llm(api_key)
        
        # Select prompt based on model/provider
        model_name_lower = self.model_name.lower()
        if ("gemma" in model_name_lower and "270" in model_name_lower) or "mcq" in model_name_lower:
            # Phase 14: Reasoning-First MCQ Prompt
            from langchain_core.prompts import PromptTemplate
            prompt = PromptTemplate.from_template(
                """Klasifikasikan berita berikut.
Pilih:
A. native ads
B. berita murni

Judul: {title}
Konten: {content}

Analisis:
"""
            )
        elif "gemma" in model_name_lower:
            # Phase 10: Standard template for larger models
            from langchain_core.prompts import PromptTemplate
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

Judul: {title}
Konten: {content}

Output (JSON):
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
                hf_token = api_key or os.environ.get("HF_TOKEN")
                
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    token=hf_token
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

                # Fix for multi-GPU device-side assert: set explicit device 0
                device_map = {"": 0} if torch.cuda.is_available() else "auto"

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
                
                # Add stop sequence for JSON block
                stop_sequences = ["}\n", "} ", tokenizer.eos_token]
                
                is_mcq = ("gemma" in model_name_lower and "270" in model_name_lower) or "mcq" in model_name_lower
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=128 if is_mcq else 256,
                    temperature=0.7,
                    do_sample=False,
                    repetition_penalty=1.0 if is_mcq else 1.1, # 1.0 is safer for tiny MCQ models
                    return_full_text=False,
                    stop_sequence=stop_sequences[0]
                )
                
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
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Classify content as native ads or pure news.
        
        Args:
            content: Main content text
            title: Article title
            summary: Content summary
            context: Retrieved context from RAG
            
        Returns:
            Classification result with label, confidence, and reasoning
        """
        try:
            logger.info("Classifying content...")
            
            # Prepare input
            input_data = {
                "title": title or content[:100],
                "content": content[:400] # Use same char limit as training/eval HF scripts
            }
            
            # Use chat template for local models to ensure instruction following
            if self.provider == "local" and self.tokenizer:
                messages = [{"role": "user", "content": self.chain.first.format(**input_data)}]
                templated_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                response = self.llm.invoke(templated_prompt)
            else:
                # Use LCEL with explicit dict
                response = self.chain.invoke(input_data)
            
            # Parse response (response is already a string thanks to StrOutputParser)
            result = self._parse_response(response)
            
            # Add metadata
            result['metadata'] = {
                'model': self.model_name,
                'provider': self.provider,
                'use_few_shot': self.use_few_shot,
                'input_length': len(content)
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
            
            # Phase 14: First check for "Jawaban: A/B" pattern in the FULL response
            # This is the expected output format for Reasoning-First MCQ models
            jawaban_match = re.search(r'Jawaban[:\s]*([AB])\b', response, re.IGNORECASE)
            if jawaban_match:
                label_code = jawaban_match.group(1).upper()
                # Extract reasoning (everything before "Jawaban:")
                reasoning = response[:jawaban_match.start()].strip()
                if not reasoning:
                    reasoning = f'Detected MCQ Label: {label_code}'
                if label_code == "A":
                    return {'label': 'native ads', 'confidence': 0.95, 'reasoning': reasoning[:300]}
                else:
                    return {'label': 'berita murni', 'confidence': 0.95, 'reasoning': reasoning[:300]}
            
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
            
            # Phase 12: Fallback for raw labels
            clean_resp_lower = response.lower().strip()
            if clean_resp_lower.startswith("native ads") or "native ads" in clean_resp_lower[:50]:
                return {'label': 'native ads', 'confidence': 0.9, 'reasoning': 'Detected via label start (fallback).'}
            elif clean_resp_lower.startswith("berita murni") or "berita murni" in clean_resp_lower[:50]:
                return {'label': 'berita murni', 'confidence': 0.9, 'reasoning': 'Detected via label start (fallback).'}
            
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
                raise json.JSONDecodeError("No JSON or label found", response, 0)
            
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
            
                
            return {
                'label': result.get('label', 'unknown'),
                'confidence': float(result.get('confidence', 0.5)),
                'reasoning': reasoning[:200]  # Truncate long reasoning
            }
                
        except (json.JSONDecodeError, Exception) as e:
            # Phase 8: If it's not JSON, it's likely a direct label from Gemma 3 (Phase 8/finetune.py style)
            clean_resp = response.lower().strip()
            
            # Priority 1: Check if the response STARTS with the label (standard Gemma/Phase 8 output)
            if clean_resp.startswith("native ads"):
                return {'label': 'native ads', 'confidence': 0.95, 'reasoning': clean_resp}
            elif clean_resp.startswith("berita murni"):
                return {'label': 'berita murni', 'confidence': 0.95, 'reasoning': clean_resp}
            
            # Priority 2: Keyword search with protection (checking for existence only in first line or as definitive mentions)
            first_line = clean_resp.split('\n')[0]
            if "native ads" in first_line:
                return {'label': 'native ads', 'confidence': 0.9, 'reasoning': clean_resp}
            elif "berita murni" in first_line:
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

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity for a given text using the local model."""
        if self.provider != "local" or not self.tokenizer:
            return 0.0
            
        try:
            import torch
            model = self.llm.pipeline.model
            tokenizer = self.tokenizer
            
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
