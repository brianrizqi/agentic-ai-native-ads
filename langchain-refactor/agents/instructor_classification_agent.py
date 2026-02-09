"""
Instructor-based Classification Agent for Native Ads Detection
Uses Instructor library for structured output with Pydantic models
Optimized for Gemma v7 model
"""

from typing import Optional, Literal, Any
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class ClassificationResult(BaseModel):
    """Structured output model for classification results."""
    
    label: Literal["native ads", "berita murni"] = Field(
        description="Classification label: 'native ads' for promotional content, 'berita murni' for pure news"
    )
    confidence: float = Field(
        ge=0.0, 
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(
        max_length=200,
        description="Brief explanation in Indonesian (max 200 characters)"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "label": "native ads",
                    "confidence": 0.85,
                    "reasoning": "Artikel mempromosikan produk dengan bahasa persuasif dan hanya satu sudut pandang positif"
                },
                {
                    "label": "berita murni",
                    "confidence": 0.92,
                    "reasoning": "Berita objektif tanpa promosi, menyajikan berbagai sudut pandang"
                }
            ]
        }


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
                self.client = instructor.from_transformers(
                    self.model,
                    tokenizer=self.tokenizer,
                    mode=instructor.Mode.JSON
                )
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
        
        prompt = f"""Klasifikasikan berita berikut sebagai "native ads" atau "berita murni".

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
Konten: {content_truncated}

Output (JSON):
{{"label": "native ads" atau "berita murni", "confidence": 0.0-1.0, "reasoning": "alasan singkat (max 150 karakter)"}}

Klasifikasi:"""
        
        return prompt
    
    def classify(
        self,
        title: str,
        content: str,
        return_raw: bool = False
    ) -> ClassificationResult:
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
                    response_model=ClassificationResult,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                logger.info(f"Classification: {result.label} (confidence: {result.confidence:.2f})")
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
                    result = ClassificationResult(**result_dict)
                    logger.info(f"Classification: {result.label} (confidence: {result.confidence:.2f})")
                    return result
                else:
                    logger.warning("No JSON found in output, using fallback")
                    return ClassificationResult(
                        label="berita murni",
                        confidence=0.5,
                        reasoning="Failed to parse model output"
                    )
                    
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return ClassificationResult(
                label="berita murni",
                confidence=0.5,
                reasoning=f"Error during classification: {str(e)[:100]}"
            )
    
    def classify_batch(
        self,
        articles: list[dict[str, str]],
        show_progress: bool = True
    ) -> list[ClassificationResult]:
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
