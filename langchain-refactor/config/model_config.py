"""
Model Configuration
Centralized configuration for LLM models and embeddings
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    
    # LLM Settings
    llm_provider: str = "openai"  # 'openai', 'openrouter', or 'huggingface'
    llm_model_name: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1000
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None  # Custom API base URL (for OpenRouter, etc.)
    
    # Embedding Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector Store Settings
    vectorstore_type: str = "faiss"  # 'faiss' or 'chroma'
    vectorstore_persist_dir: str = "data/vectorstore"
    
    # RAG Settings
    use_rag: bool = True
    rag_top_k: int = 5
    
    # Classification Settings
    use_few_shot: bool = True
    confidence_threshold: float = 0.7
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'llm_provider': self.llm_provider,
            'llm_model_name': self.llm_model_name,
            'llm_temperature': self.llm_temperature,
            'llm_max_tokens': self.llm_max_tokens,
            'embedding_model': self.embedding_model,
            'vectorstore_type': self.vectorstore_type,
            'vectorstore_persist_dir': self.vectorstore_persist_dir,
            'use_rag': self.use_rag,
            'rag_top_k': self.rag_top_k,
            'use_few_shot': self.use_few_shot,
            'confidence_threshold': self.confidence_threshold
        }


# Predefined configurations
OPENAI_GPT35_CONFIG = ModelConfig(
    llm_provider="openai",
    llm_model_name="gpt-3.5-turbo",
    llm_temperature=0.3
)

OPENAI_GPT4_CONFIG = ModelConfig(
    llm_provider="openai",
    llm_model_name="gpt-4",
    llm_temperature=0.3
)

HUGGINGFACE_MISTRAL_CONFIG = ModelConfig(
    llm_provider="huggingface",
    llm_model_name="mistralai/Mistral-7B-Instruct-v0.2",
    llm_temperature=0.3
)

HUGGINGFACE_LLAMA_CONFIG = ModelConfig(
    llm_provider="huggingface",
    llm_model_name="meta-llama/Llama-2-7b-chat-hf",
    llm_temperature=0.3
)

OPENROUTER_GPT4O_MINI_CONFIG = ModelConfig(
    llm_provider="openrouter",
    llm_model_name="openai/gpt-4o-mini",
    llm_temperature=0.3,
    llm_base_url="https://openrouter.ai/api/v1"
)

# Default configuration
DEFAULT_CONFIG = OPENAI_GPT35_CONFIG
