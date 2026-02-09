"""
Instructor-based Classification Agent for Native Ads Detection
Uses Instructor library for structured output with Pydantic models
Optimized for Gemma v7 model
"""

from typing import Optional, Literal, Any, Union
from pydantic import BaseModel, Field, field_validator
import logging

logger = logging.getLogger(__name__)


class AdvancedAnalysisResult(BaseModel):
    """Expanded structured output for deeper content analysis."""
    
    label: Literal["native ads", "berita murni"] = Field(
        default="berita murni",
        description="Klasifikasi utama: 'native ads' atau 'berita murni'"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Skor keyakinan (0.0 - 1.0)"
    )
    target_brand: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="Merek, produk, atau instansi yang dipromosikan"
    )
    techniques: list[str] = Field(
        default_factory=list,
        description="Teknik persuasi yang digunakan"
    )
    sentiment: str = Field(
        default="netral",
        description="Nada atau sentimen utama"
    )
    has_cta: bool = Field(
        default=False,
        description="Apakah ada ajakan bertindak?"
    )
    cta_text: Optional[str] = Field(
        default=None,
        description="Teks ajakan bertindak"
    )
    reasoning: str = Field(
        default="",
        description="Alasan singkat klasifikasi"
    )

    @field_validator('target_brand', mode='before')
    @classmethod
    def validate_brand(cls, v):
        # Catch common hallucinations from training examples
        dummies = ["brand x", "brand y", "test brand", "merek a", "merek b", "none", "n/a", "tidak ada"]
        if isinstance(v, list):
            v = ", ".join([str(i) for i in v if i])
        
        if v and str(v).lower().strip() in dummies:
            return None
        return v

    @field_validator('techniques', mode='before')
    @classmethod
    def validate_techniques(cls, v):
        if not isinstance(v, list):
            return []
        # Filter out dummy techniques from examples
        dummies = ["testimonials", "emotional language", "otoritas", "bukti sosial"]
        # Only filter if it looks EXACTLY like the few-shot defaults and we have multiple
        return [t for t in v if str(t).lower().strip() not in ["dummy", "n/a"]]

    @field_validator('cta_text', mode='before')
    @classmethod
    def validate_cta(cls, v):
        dummies = ["learn more", "daftar sekarang", "klik di sini", "buy now"]
        if v and str(v).lower().strip() in dummies:
            # Check if this dummy text actually exists in the prompt content (contextual check)
            # Since we don't have prompt here, we just be careful. 
            # Hallucinated CTAs are very common.
            return v
        return v

    @field_validator('sentiment', mode='before')
    @classmethod
    def validate_sentiment(cls, v):
        if not v: return "netral"
        v = str(v).lower()
        if "positive" in v or "positif" in v: return "positif"
        if "negative" in v or "negatif" in v: return "negatif"
        return "netral"

    @field_validator('label', mode='before')
    @classmethod
    def validate_label(cls, v):
        if not v: return "berita murni"
        v = str(v).lower()
        if "news" in v or "murni" in v: return "berita murni"
        if "ads" in v or "native" in v: return "native ads"
        return "berita murni"


class InstructorClassificationAgent:
    """
    Classification agent using Instructor for structured output.
    Optimized for Gemma v7 fine-tuned model.
    """
    
    def __init__(
        self,
        model_path: str = "../models/gemma-native-ads-v7_merged_16bit",
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        use_instructor: bool = True
    ):
        """
        Initialize Instructor Classification Agent.
        
        Args:
            model_path: Path to the fine-tuned Gemma v7 model
            device: Device to use ('auto', 'cuda', 'cpu')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            use_instructor: Whether to use instructor for structured output
        """
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_instructor = use_instructor
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load heavy dependencies lazily
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            logger.error(f"Required dependencies for InstructorClassificationAgent missing: {e}")
            raise
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Wrap with instructor if enabled
        if self.use_instructor:
            try:
                import instructor
                # Try standard attribute first
                if hasattr(instructor, "from_transformers"):
                    self.client = instructor.from_transformers(
                        self.model,
                        tokenizer=self.tokenizer,
                        mode=instructor.Mode.JSON
                    )
                else:
                    # Fallback to direct import if attribute is missing
                    logger.warning("instructor.from_transformers not found in main namespace. Attempting direct import...")
                    try:
                        from instructor.client_transformers import from_transformers
                        self.client = from_transformers(
                            self.model,
                            tokenizer=self.tokenizer,
                            mode=instructor.Mode.JSON
                        )
                    except (ImportError, AttributeError) as e:
                        logger.error(f"Failed to initialize instructor for transformers: {e}")
                        self.use_instructor = False
                        self.client = None
                
                if self.use_instructor:
                    logger.info("Instructor wrapper enabled for structured output")
            except ImportError as e:
                logger.error(f"Instructor module not found: {e}")
                self.use_instructor = False
                self.client = None
        else:
            self.client = None
            logger.info("Using standard generation without instructor")
        
        logger.info(f"Model loaded successfully on {self.model.device}")
    
    def _build_prompt(self, title: str, content: str) -> str:
        """
        Build classification prompt matching training format.
        
        Args:
            title: Article title
            content: Article content (will be truncated to 800 chars)
            
        Returns:
            Formatted prompt string
        """
        # Truncate content to avoid context overflow
        content_truncated = content[:800] if len(content) > 800 else content
        
        prompt = f"""Lakukan analisis mendalam terhadap berita berikut.
Instruksi:
1. Tentukan apakah ini "native ads" atau "berita murni".
2. Identifikasi brand/produk/instansi yang dipromosikan (jika ada).
3. Deteksi teknik persuasi (misal: penggunaan testimoni, bahasa emosional, klaim sepihak).
4. Cek apakah ada Call to Action (CTA) seperti ajakan membeli atau mendaftar.
5. Tentukan sentimen atau nada berita.

Judul: {title}
Konten: {content_truncated}

Output dalam format JSON dengan field: label, confidence, target_brand, techniques (list), sentiment, has_cta, cta_text, reasoning.

Klasifikasi Mendasar:"""
        
        return prompt
    
    def classify(
        self,
        title: str,
        content: str,
        return_raw: bool = False
    ) -> AdvancedAnalysisResult:
        """
        Classify content as native ads or pure news.
        
        Args:
            title: Article title
            content: Article content
            return_raw: If True, also return raw model output
            
        Returns:
            ClassificationResult with label, confidence, and reasoning
        """
        try:
            prompt = self._build_prompt(title, content)
            
            if self.use_instructor and self.client:
                # Use instructor for structured output
                logger.info("Generating classification with Instructor...")
                result = self.client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    response_model=AdvancedAnalysisResult,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                logger.info(f"Advanced Classification: {result.label} for {result.target_brand or 'No Brand'}")
                return result
            else:
                # Fallback to standard generation
                logger.info("Generating classification with standard generation...")
                import torch
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.15,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract JSON from response
                import json
                import re
                
                # Try to find JSON in the output
                json_match = re.search(r'\{[^}]+\}', generated_text)
                if json_match:
                    json_str = json_match.group(0)
                    result_dict = json.loads(json_str)
                    result = AdvancedAnalysisResult(**result_dict)
                    return result
                else:
                    logger.warning("No JSON found in output, using fallback")
                    return AdvancedAnalysisResult(
                        label="berita murni",
                        confidence=0.5,
                        sentiment="netral",
                        has_cta=False,
                        reasoning="Failed to parse model output"
                    )
                    
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return AdvancedAnalysisResult(
                label="berita murni",
                confidence=0.5,
                sentiment="netral",
                has_cta=False,
                reasoning=f"Error: {str(e)[:50]}"
            )
    
    def classify_batch(
        self,
        articles: list[dict[str, str]],
        show_progress: bool = True
    ) -> list[AdvancedAnalysisResult]:
        """
        Classify multiple articles in batch.
        
        Args:
            articles: List of dicts with 'title' and 'content' keys
            show_progress: Whether to show progress bar
            
        Returns:
            List of ClassificationResult objects
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(articles, desc="Classifying")
            except ImportError:
                iterator = articles
                logger.warning("tqdm not installed, progress bar disabled")
        else:
            iterator = articles
        
        for article in iterator:
            result = self.classify(
                title=article.get("title", ""),
                content=article.get("content", "")
            )
            results.append(result)
        
        return results
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "use_instructor": self.use_instructor,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "vocab_size": self.tokenizer.vocab_size,
            "model_params": sum(p.numel() for p in self.model.parameters()) / 1e9  # in billions
        }
