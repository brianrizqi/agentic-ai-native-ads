# LangChain Refactor - Quick Start Guide

## Installation

```bash
cd langchain-refactor
pip install -r requirements.txt
```

## Setup Vector Store (for RAG)

```bash
# Create vector store from training dataset
python setup_vectorstore.py \
  --dataset ../data/llm_dataset_instruction.json \
  --output-dir data/vectorstore
```

## Usage

### 1. Single URL Classification

```bash
python main.py \
  --url "https://example.com/article" \
  --model gpt-3.5-turbo \
  --api-key YOUR_OPENAI_KEY
```

### 2. Batch Processing

```bash
# Create urls.txt with one URL per line
python main.py \
  --urls-file urls.txt \
  --output results.json \
  --api-key YOUR_OPENAI_KEY
```

### 3. Using HuggingFace Models

```bash
python main.py \
  --url "https://example.com" \
  --provider huggingface \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --api-key YOUR_HF_TOKEN
```

### 4. Using OpenRouter (Recommended for Cost-Effective GPT-4)

```bash
# Using OpenRouter with GPT-4o-mini
python main.py \
  --url "https://example.com/article" \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --api-key YOUR_OPENROUTER_KEY

# Batch processing with OpenRouter
python main.py \
  --urls-file urls.txt \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --output results.json \
  --api-key YOUR_OPENROUTER_KEY
```

**Get OpenRouter API Key:** https://openrouter.ai/keys

### 5. Programmatic Usage

```python
from chains.full_pipeline_chain import FullPipelineChain

# Initialize pipeline
pipeline = FullPipelineChain(
    model_name="gpt-3.5-turbo",
    provider="openai",
    api_key="your-key"
)

# Or use OpenRouter
pipeline_openrouter = FullPipelineChain(
    model_name="openai/gpt-4o-mini",
    provider="openrouter",
    api_key="your-openrouter-key"
)

# Classify URL
result = pipeline.run("https://example.com/article")

print(f"Label: {result['classification']['label']}")
print(f"Confidence: {result['classification']['confidence']:.2f}")
```

## Configuration

Edit `config/model_config.py` to customize:
- LLM model and provider
- Embedding model
- Vector store settings
- RAG parameters

### 5. Automated Dataset Collection

You can now automatically collect news articles, classify them, and save them as a dataset:

```bash
python main.py --collect 50 --provider openrouter --model openai/gpt-4o-mini
```

This will:
1. Crawl latest news from Indonesian portals.
2. Clean, summarize, and classify each article.
3. Save the results to `data/news_dataset.csv` and `data/news_dataset_llm_ready.json`.

## Project Structure

```
langchain-refactor/
├── agents/          # LangChain agents
├── chains/          # Workflow chains
├── tools/           # Individual tools
├── prompts/         # Prompt templates
├── vector_stores/   # FAISS vector store
├── config/          # Configuration
├── utils/           # Utilities
├── main.py          # CLI entry point
└── examples.py      # Usage examples
```

## Features

✅ Modular architecture with LangChain
✅ Multiple LLM providers (OpenAI, OpenRouter, HuggingFace)
✅ RAG with FAISS vector store
✅ Few-shot learning prompts
✅ Batch processing
✅ CLI and programmatic interfaces
✅ Comprehensive logging

## Next Steps

1. Setup vector store: `python setup_vectorstore.py`
2. Test with example: `python examples.py`
3. Run on your URLs: `python main.py --url YOUR_URL`

See `examples.py` for more usage patterns!
