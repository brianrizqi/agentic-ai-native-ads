"""
Example: Batch URL Processing
Contoh untuk memproses multiple URLs sekaligus
"""

from main import AgenticAISystem
import json
from pathlib import Path
import pandas as pd


def main():
    # Load configuration
    config_path = Path(__file__).parent / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize system
    print("Initializing Agentic AI System...")
    system = AgenticAISystem(config)
    
    # Add knowledge base
    print("Adding knowledge base documents...")
    knowledge_docs = [
        "Technology news covers innovations in software, hardware, and digital services.",
        "Business articles discuss market trends, company performance, and economic indicators.",
        "Health and medical content focuses on wellness, diseases, and treatment options.",
        "Education content includes learning resources, teaching methods, and academic research.",
        "Entertainment news covers movies, music, celebrities, and cultural events."
    ]
    system.add_knowledge_base_documents(knowledge_docs)
    
    # URLs to process
    urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://en.wikipedia.org/wiki/Computer_vision"
    ]
    
    print(f"\nProcessing {len(urls)} URLs...")
    print("="*80)
    
    # Process all URLs
    all_results = system.process_multiple_urls(urls)
    
    # Compile results for analysis
    summary_data = []
    
    for result in all_results:
        if result['success']:
            summary_data.append({
                'URL': result['url'],
                'Title': result['scraped_data']['title'][:50],
                'Word Count': result['scraped_data']['word_count'],
                'Classification': result['classification']['label'],
                'Confidence': f"{result['classification']['confidence']:.2%}",
                'Quality': result['feedback_analysis']['quality_level'],
                'Retrieved Docs': result['retrieved_context']['num_documents']
            })
    
    # Display summary table
    if summary_data:
        df = pd.DataFrame(summary_data)
        print("\n" + "="*80)
        print("BATCH PROCESSING SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
    
    # Detailed results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    for i, result in enumerate(all_results, 1):
        print(f"\n[{i}] {result['url']}")
        print("-" * 80)
        
        if result['success']:
            print(f"Title: {result['scraped_data']['title']}")
            print(f"Classification: {result['classification']['label']} ({result['classification']['confidence']:.2%})")
            print(f"Explanation: {result['explanation']['summary'][:150]}...")
        else:
            print(f"Error: {result['error']}")
    
    # Export results
    output_file = 'data/batch_results.json'
    Path('data').mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results exported to: {output_file}")
    
    # System statistics
    print("\n" + "="*80)
    print("SYSTEM STATISTICS")
    print("="*80)
    stats = system.get_system_stats()
    print(f"Total Documents in KB: {stats['retriever_stats']['total_documents']}")
    print(f"Total Feedback Collected: {stats['feedback_summary'].get('total_feedback', 0)}")


if __name__ == "__main__":
    main()
