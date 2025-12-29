"""
Example: Simple URL Processing
Contoh sederhana untuk memproses satu URL
"""

from main import AgenticAISystem
import json
from pathlib import Path


def main():
    # Load configuration
    config_path = Path(__file__).parent / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize system
    print("Initializing Agentic AI System...")
    system = AgenticAISystem(config)
    
    # Add sample knowledge base
    print("\nAdding sample documents to knowledge base...")
    sample_docs = [
        "Artificial Intelligence (AI) is revolutionizing healthcare through improved diagnostics and personalized treatment plans.",
        "Machine learning algorithms require large datasets for training and validation to achieve high accuracy.",
        "Natural language processing enables computers to understand, interpret, and generate human language.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns in data.",
        "Computer vision allows machines to interpret and understand visual information from the world."
    ]
    system.add_knowledge_base_documents(sample_docs)
    
    # Process a URL
    url = input("\nEnter URL to process (or press Enter for example): ").strip()
    if not url:
        url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    print(f"\nProcessing: {url}")
    print("="*80)
    
    results = system.process_url(url)
    
    # Display results
    if results['success']:
        print("\n✓ Processing successful!")
        print("\n--- SCRAPED DATA ---")
        print(f"Title: {results['scraped_data']['title']}")
        print(f"Word Count: {results['scraped_data']['word_count']}")
        
        print("\n--- PREPROCESSING ---")
        print(f"Cleaned Text Length: {results['preprocessed_data']['cleaned_text_length']} chars")
        print(f"Token Count: {results['preprocessed_data']['token_count']}")
        print(f"Summary: {results['preprocessed_data']['summary'][:200]}...")
        
        print("\n--- RETRIEVAL ---")
        print(f"Retrieved Documents: {results['retrieved_context']['num_documents']}")
        for i, doc in enumerate(results['retrieved_context']['documents'][:3], 1):
            print(f"\n  Document {i} (Score: {doc['relevance_score']:.2f}):")
            print(f"  {doc['content'][:100]}...")
        
        print("\n--- CLASSIFICATION ---")
        print(f"Label: {results['classification']['label']}")
        print(f"Confidence: {results['classification']['confidence']:.2%}")
        print(f"Reasoning: {results['classification'].get('reasoning', 'N/A')}")
        
        print("\n--- EXPLANATION ---")
        print(f"Summary: {results['explanation']['summary']}")
        print(f"\nKey Factors:")
        for factor in results['explanation'].get('key_factors', [])[:5]:
            print(f"  • {factor}")
        
        print("\n--- FEEDBACK ANALYSIS ---")
        print(f"Quality Level: {results['feedback_analysis']['quality_level']}")
        print(f"Quality Score: {results['feedback_analysis']['quality_score']:.2f}")
        print(f"Needs Improvement: {results['feedback_analysis']['needs_improvement']}")
        
    else:
        print(f"\n✗ Error: {results['error']}")
    
    # System stats
    print("\n" + "="*80)
    print("SYSTEM STATISTICS")
    print("="*80)
    stats = system.get_system_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
