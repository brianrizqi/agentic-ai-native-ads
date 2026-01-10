"""
Example Usage Scripts for LangChain Refactored System
"""

# Example 1: Simple URL Classification
print("="*80)
print("Example 1: Classify Single URL")
print("="*80)

from chains.full_pipeline_chain import FullPipelineChain

# Initialize pipeline
pipeline = FullPipelineChain(
    model_name="gpt-3.5-turbo",
    provider="openai",
    api_key="your-api-key-here"
)

# Classify a URL
result = pipeline.run("https://example.com/article")

print(f"Classification: {result['classification']['label']}")
print(f"Confidence: {result['classification']['confidence']:.2f}")
print(f"Reasoning: {result['classification']['reasoning']}")


# Example 2: Batch Processing
print("\n" + "="*80)
print("Example 2: Batch Process Multiple URLs")
print("="*80)

urls = [
    "https://example.com/article1",
    "https://example.com/article2",
    "https://example.com/article3"
]

results = pipeline.run_batch(urls)

for i, result in enumerate(results, 1):
    print(f"\n[{i}] {result['url']}")
    print(f"    Label: {result['classification']['label']}")
    print(f"    Confidence: {result['classification']['confidence']:.2f}")


# Example 3: Using Individual Tools
print("\n" + "="*80)
print("Example 3: Using Individual Tools")
print("="*80)

from tools import WebScraperTool, TextCleanerTool, FeatureExtractorTool

# Scrape content
scraper = WebScraperTool()
scraped = scraper.run("https://example.com")

# Clean text
cleaner = TextCleanerTool()
cleaned_text = cleaner.run(scraped['text'])

# Extract features
extractor = FeatureExtractorTool()
features = extractor.run(cleaned_text)

print(f"Word count: {features['word_count']}")
print(f"Sentence count: {features['sentence_count']}")
print(f"Lexical diversity: {features['lexical_diversity']:.2f}")


# Example 4: Using Classification Agent Directly
print("\n" + "="*80)
print("Example 4: Direct Classification Agent Usage")
print("="*80)

from agents.classification_agent import ClassificationAgent

# Initialize agent
agent = ClassificationAgent(
    model_name="gpt-3.5-turbo",
    provider="openai",
    use_few_shot=True
)

# Classify content
content = "Promo spesial hari ini! Dapatkan diskon 50% untuk semua produk."
result = agent.classify(
    content=content,
    title="Promo Spesial Hari Ini"
)

print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Reasoning: {result['reasoning']}")


# Example 5: With Vector Store (RAG)
print("\n" + "="*80)
print("Example 5: Using Vector Store for RAG")
print("="*80)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Create sample documents
docs = [
    Document(page_content="Native ads adalah iklan berbayar yang menyerupai konten editorial."),
    Document(page_content="Berita murni adalah konten jurnalistik tanpa promosi komersial."),
]

# Create vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# Initialize pipeline with vector store
pipeline_with_rag = FullPipelineChain(
    model_name="gpt-3.5-turbo",
    provider="openai",
    vectorstore=vectorstore
)

# Run classification with RAG
result = pipeline_with_rag.run("https://example.com/article")

print(f"Classification: {result['classification']['label']}")
print(f"Has context: {result['metadata']['has_context']}")


# Example 6: Using OpenRouter Provider
print("\n" + "="*80)
print("Example 6: Using OpenRouter with GPT-4o-mini")
print("="*80)

from chains.full_pipeline_chain import FullPipelineChain

# Initialize pipeline with OpenRouter
pipeline_openrouter = FullPipelineChain(
    model_name="openai/gpt-4o-mini",
    provider="openrouter",
    api_key="your-openrouter-api-key"  # Or use environment variable
)

# Classify content
result = pipeline_openrouter.run("https://example.com/article")

print(f"Classification: {result['classification']['label']}")
print(f"Confidence: {result['classification']['confidence']:.2f}")
print(f"Model: openai/gpt-4o-mini via OpenRouter")

