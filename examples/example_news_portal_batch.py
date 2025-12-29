"""
Example: Batch Processing Portal Berita
Contoh untuk scraping dan analisis batch dari portal berita Indonesia
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from main import AgenticAISystem
import json
import pandas as pd
from datetime import datetime


def main():
    """
    Batch processing artikel dari berbagai portal berita untuk penelitian.
    """
    print("\n" + "="*80)
    print("BATCH PROCESSING - PORTAL BERITA INDONESIA")
    print("Native Ads Detection Research")
    print("="*80 + "\n")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize system
    print("Initializing system...")
    system = AgenticAISystem(config)
    
    # Setup knowledge base
    print("Setting up knowledge base...")
    knowledge_docs = [
        "Native advertising menyerupai konten editorial dengan tujuan promosi tersembunyi.",
        "Ciri native ads: bahasa promosi, brand mentions, call-to-action, tanpa disclosure.",
        "Editorial content: objektif, faktual, multiple sources, balanced perspective.",
        "Sponsored content memiliki label jelas: 'Sponsored', 'Advertorial', 'Konten Bersponsor'.",
        "Portal berita Indonesia: Detik, Kompas, Tribun, CNN Indonesia, Tempo, dll."
    ]
    system.add_knowledge_base_documents(knowledge_docs)
    
    # Sample URLs dari berbagai portal berita
    # NOTE: Ganti dengan URL artikel asli untuk penelitian
    sample_urls = [
        # Detik.com
        "https://www.detik.com/example-article-1",
        "https://www.detik.com/example-article-2",
        
        # Kompas.com
        "https://www.kompas.com/example-article-1",
        "https://www.kompas.com/example-article-2",
        
        # Tribunnews.com
        "https://www.tribunnews.com/example-article-1",
    ]
    
    print(f"\nProcessing {len(sample_urls)} articles...")
    print("="*80 + "\n")
    
    # Process all URLs
    all_results = []
    for i, url in enumerate(sample_urls, 1):
        print(f"[{i}/{len(sample_urls)}] Processing: {url}")
        result = system.process_url(url)
        all_results.append(result)
        
        if result['success']:
            label = result['classification']['label']
            confidence = result['classification']['confidence']
            print(f"  → {label} ({confidence:.2%})")
        else:
            print(f"  → Error: {result['error']}")
        print()
    
    # Compile results
    successful_results = [r for r in all_results if r['success']]
    
    if not successful_results:
        print("No successful results to analyze.")
        return
    
    # Create summary DataFrame
    summary_data = []
    for result in successful_results:
        summary_data.append({
            'URL': result['url'],
            'Portal': result['url'].split('/')[2] if '/' in result['url'] else 'Unknown',
            'Title': result['scraped_data']['title'][:50] + '...',
            'Word Count': result['scraped_data']['word_count'],
            'Classification': result['classification']['label'],
            'Confidence': f"{result['classification']['confidence']:.2%}",
            'Quality': result['feedback_analysis']['quality_level']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Display summary
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80 + "\n")
    print(df.to_string(index=False))
    
    # Statistics
    print("\n" + "="*80)
    print("CLASSIFICATION STATISTICS")
    print("="*80 + "\n")
    
    classification_counts = df['Classification'].value_counts()
    print("Distribution:")
    for label, count in classification_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    print(f"\nAverage Confidence: {df['Confidence'].str.rstrip('%').astype(float).mean():.2f}%")
    
    # Quality analysis
    print("\n" + "="*80)
    print("QUALITY ANALYSIS")
    print("="*80 + "\n")
    
    quality_counts = df['Quality'].value_counts()
    print("Quality Distribution:")
    for quality, count in quality_counts.items():
        print(f"  {quality}: {count}")
    
    # Export results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Export to JSON
    json_output = f'data/batch_results_{timestamp}.json'
    Path('data').mkdir(exist_ok=True)
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Full results exported to: {json_output}")
    
    # Export to CSV
    csv_output = f'data/batch_summary_{timestamp}.csv'
    df.to_csv(csv_output, index=False, encoding='utf-8')
    print(f"✓ Summary exported to: {csv_output}")
    
    # Export training data
    training_output = f'data/training_data_{timestamp}.json'
    system.feedback_agent.export_training_data(training_output)
    print(f"✓ Training data exported to: {training_output}")
    
    # System stats
    print("\n" + "="*80)
    print("SYSTEM STATISTICS")
    print("="*80 + "\n")
    
    stats = system.get_system_stats()
    print(f"Knowledge Base Documents: {stats['retriever_stats']['total_documents']}")
    print(f"Total Feedback Collected: {stats['feedback_summary'].get('total_feedback', 0)}")
    
    print("\n" + "="*80)
    print("Batch processing complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
