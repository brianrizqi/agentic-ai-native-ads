"""
Main Entry Point for LangChain Refactored System
"""

import argparse
import logging
import json
from pathlib import Path

from chains.full_pipeline_chain import FullPipelineChain

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Native Ads Detection System (LangChain Refactored)'
    )
    parser.add_argument('--url', type=str, help='URL to analyze')
    parser.add_argument('--urls-file', type=str, help='File containing URLs (one per line)')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                       help='LLM model name')
    parser.add_argument('--provider', type=str, default='openai',
                       choices=['openai', 'openrouter', 'huggingface'],
                       help='LLM provider')
    parser.add_argument('--api-key', type=str, help='API key for LLM provider')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = FullPipelineChain(
        model_name=args.model,
        provider=args.provider,
        api_key=args.api_key
    )
    
    # Process URLs
    results = []
    
    if args.url:
        # Single URL
        logger.info(f"Processing single URL: {args.url}")
        result = pipeline.run(args.url)
        results.append(result)
        
    elif args.urls_file:
        # Multiple URLs from file
        logger.info(f"Processing URLs from file: {args.urls_file}")
        with open(args.urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        results = pipeline.run_batch(urls)
    
    else:
        parser.print_help()
        return
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")
    
    for result in results:
        print(f"URL: {result.get('url')}")
        print(f"Title: {result.get('title', 'N/A')}")
        
        if result.get('status') == 'success':
            classification = result.get('classification', {})
            print(f"Classification: {classification.get('label')}")
            print(f"Confidence: {classification.get('confidence', 0):.2f}")
            print(f"Reasoning: {classification.get('reasoning', 'N/A')}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        print("-" * 80 + "\n")
    
    # Save to file if specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
