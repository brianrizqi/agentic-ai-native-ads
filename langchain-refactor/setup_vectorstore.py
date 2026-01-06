"""
Script to setup vector store from training dataset
"""

import argparse
import logging
from pathlib import Path

from vector_stores import NativeAdsVectorStore
from utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Setup Vector Store for RAG')
    parser.add_argument(
        '--dataset',
        type=str,
        default='../data/llm_dataset_instruction.json',
        help='Path to training dataset JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/vectorstore',
        help='Directory to save vector store'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='HuggingFace embedding model name'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    
    logger.info("="*80)
    logger.info("VECTOR STORE SETUP")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Embedding Model: {args.embedding_model}")
    logger.info("")
    
    # Check dataset exists
    if not Path(args.dataset).exists():
        logger.error(f"Dataset not found: {args.dataset}")
        return
    
    # Create vector store
    logger.info("Initializing vector store...")
    vectorstore = NativeAdsVectorStore(
        embedding_model=args.embedding_model,
        persist_directory=args.output_dir
    )
    
    # Load dataset and create index
    logger.info("Loading dataset and creating vector index...")
    vectorstore.create_from_dataset(args.dataset)
    
    # Get stats
    stats = vectorstore.get_stats()
    logger.info("")
    logger.info("Vector Store Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("")
    logger.info("="*80)
    logger.info("✅ Vector store setup complete!")
    logger.info(f"Saved to: {args.output_dir}")
    logger.info("="*80)
    
    # Test search
    logger.info("\nTesting vector search...")
    test_query = "Promo diskon produk"
    results = vectorstore.similarity_search(test_query, k=3)
    
    logger.info(f"Query: '{test_query}'")
    logger.info(f"Found {len(results)} results:")
    for i, doc in enumerate(results, 1):
        logger.info(f"\n[{i}] Label: {doc.metadata.get('label')}")
        logger.info(f"    Content: {doc.page_content[:100]}...")


if __name__ == "__main__":
    main()
