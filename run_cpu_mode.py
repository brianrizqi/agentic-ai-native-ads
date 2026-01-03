"""
Local Agentic AI Demo - CPU Mode
Force CPU mode untuk menghindari CUDA/MIG issues
"""

import sys
import logging
from pathlib import Path
import json
import argparse
import os

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Add directory to path
sys.path.append(str(Path(__file__).parent))

from agents.web_agent import WebAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.retriever_agent import RetrieverAgent
from agents.llm_classifier_agent import LLMClassifierAgent
from agents.explanation_agent import ExplanationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run Agentic AI Demo in CPU Mode')
    parser.add_argument('--model', type=str, default='microsoft/phi-2', 
                       help='Model name (default: microsoft/phi-2 - smaller model for CPU)')
    parser.add_argument('--url', type=str, required=True, help='URL to analyze')
    args = parser.parse_args()
    
    print("="*80)
    print(f"AGENTIC AI DEMO - CPU MODE")
    print(f"Model: {args.model}")
    print(f"Target URL: {args.url}")
    print(f"Note: Running on CPU (slower but stable)")
    print("="*80 + "\n")
    
    try:
        # 1. Initialize Agents
        print("[1/5] Initializing Agents...")
        
        web_agent = WebAgent()
        preprocess_agent = PreprocessingAgent()
        
        # Load QnA dataset for retrieval context
        dataset_path = 'data/llm_dataset_qna.json'
        if not Path(dataset_path).exists():
            print(f"[X] Dataset not found: {dataset_path}")
            print("[!] Run: python tools/dataset_converter.py first")
            return
            
        retriever_agent = RetrieverAgent(
            vector_db_path=dataset_path,
            embedding_model='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Use smaller model for CPU
        classifier_agent = LLMClassifierAgent(
            provider='huggingface',
            model_name=args.model
        )
        
        explanation_agent = ExplanationAgent(
            provider='huggingface',
            model_name=args.model
        )
        
        # 2. Web Scraping
        print("\n[2/5] Scraping Content...")
        scraped_data = web_agent.scrape(args.url)
        print(f"[OK] Title: {scraped_data['title']}")
        print(f"[OK] Length: {len(scraped_data['text'])} chars")
        
        # 3. Preprocessing
        print("\n[3/5] Preprocessing...")
        preprocessed_data = preprocess_agent.process(scraped_data)
        print(f"[OK] Cleaned Length: {len(preprocessed_data['cleaned_text'])} chars")
        print(f"[OK] Tokens: {len(preprocessed_data['tokens'])}")
        
        # 4. Retrieval
        print("\n[4/5] Retrieving Context...")
        context = retriever_agent.retrieve(preprocessed_data)
        print(f"[OK] Retrieved {len(context)} relevant documents")
        for i, doc in enumerate(context, 1):
            print(f"  - Doc {i}: Score {doc['relevance_score']:.4f}")
            
        # 5. Classification & Explanation
        print("\n[5/5] Classifying & Explaining (CPU mode - may take a while)...")
        
        # Classify
        classification = classifier_agent.classify(preprocessed_data, context)
        print(f"\n--- CLASSIFICATION RESULT ---")
        print(f"Label: {classification['label']}")
        print(f"Confidence: {classification['confidence']:.2%}")
        print(f"Reasoning: {classification['reasoning']}")
        
        # Explain
        explanation = explanation_agent.explain(classification, preprocessed_data, context)
        print(f"\n--- EXPLANATION ---")
        print(explanation['detailed_explanation'])
        
        # Save results
        output = {
            'url': args.url,
            'scraped': scraped_data['title'],
            'classification': classification,
            'explanation': explanation,
            'mode': 'CPU'
        }
        
        with open('local_demo_result_cpu.json', 'w') as f:
            json.dump(output, f, indent=2)
            
        print("\n[OK] Results saved to local_demo_result_cpu.json")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
