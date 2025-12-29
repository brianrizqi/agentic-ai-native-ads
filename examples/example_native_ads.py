"""
Example: Native Ads Detection
Contoh khusus untuk deteksi native advertising pada portal berita
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from main import AgenticAISystem
import json


def setup_native_ads_knowledge_base(system):
    """
    Setup knowledge base dengan contoh native ads dan editorial content.
    """
    print("Setting up Native Ads Knowledge Base...")
    
    # Contoh karakteristik native ads
    native_ads_examples = [
        """Native advertising adalah iklan yang dirancang menyerupai konten editorial. 
        Ciri-ciri: bahasa promosi tersembunyi, brand mentions berulang, call-to-action terselubung, 
        tidak ada disclosure statement yang jelas.""",
        
        """Indikator native ads: penggunaan kata-kata seperti 'terbaik', 'solusi', 'rekomendasi', 
        'wajib dicoba', fokus pada produk/brand tertentu, struktur artikel yang promosional.""",
        
        """Editorial content murni: fokus pada informasi faktual, objektif, tidak ada agenda promosi, 
        multiple sources, balanced perspective, jurnalisme investigatif.""",
        
        """Sponsored content biasanya memiliki label 'Sponsored', 'Advertorial', atau 'Konten Bersponsor' 
        di bagian atas artikel. Native ads sering tidak memiliki label yang jelas.""",
        
        """Perbedaan native ads vs editorial: Native ads memiliki tujuan komersial tersembunyi, 
        sementara editorial fokus pada informasi publik dan kepentingan pembaca."""
    ]
    
    system.add_knowledge_base_documents(native_ads_examples)
    print(f"✓ Added {len(native_ads_examples)} knowledge base documents\n")


def analyze_article(system, url, expected_label=None):
    """
    Analisis artikel untuk deteksi native ads.
    """
    print("="*80)
    print(f"Analyzing: {url}")
    print("="*80)
    
    result = system.process_url(url)
    
    if not result['success']:
        print(f"✗ Error: {result['error']}\n")
        return
    
    # Display results
    print("\n📰 ARTICLE INFO")
    print(f"Title: {result['scraped_data']['title']}")
    print(f"Word Count: {result['scraped_data']['word_count']}")
    
    print("\n🔍 CLASSIFICATION")
    classification = result['classification']
    print(f"Label: {classification['label']}")
    print(f"Confidence: {classification['confidence']:.2%}")
    
    if expected_label:
        is_correct = classification['label'].lower() == expected_label.lower()
        print(f"Expected: {expected_label} {'✓' if is_correct else '✗'}")
    
    print(f"\n💡 REASONING")
    print(classification.get('reasoning', 'N/A'))
    
    print(f"\n📊 EXPLANATION")
    explanation = result['explanation']
    print(f"Summary: {explanation['summary']}")
    
    print(f"\n🎯 KEY FACTORS")
    for i, factor in enumerate(explanation.get('key_factors', [])[:5], 1):
        print(f"  {i}. {factor}")
    
    print(f"\n📈 QUALITY METRICS")
    feedback = result['feedback_analysis']
    print(f"Quality Level: {feedback['quality_level']}")
    print(f"Quality Score: {feedback['quality_score']:.2f}")
    print(f"Needs Improvement: {feedback['needs_improvement']}")
    
    print(f"\n🔄 RETRIEVED CONTEXT")
    print(f"Documents Retrieved: {result['retrieved_context']['num_documents']}")
    for i, doc in enumerate(result['retrieved_context']['documents'][:2], 1):
        print(f"\n  Context {i} (Relevance: {doc['relevance_score']:.2f}):")
        print(f"  {doc['content'][:150]}...")
    
    print("\n")


def main():
    """
    Main function untuk demo native ads detection.
    """
    print("\n" + "="*80)
    print("AGENTIC AI - NATIVE ADS DETECTION SYSTEM")
    print("Penelitian Postdoc: Deteksi Native Ads pada Portal Berita Elektronik")
    print("="*80 + "\n")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize system
    print("Initializing Agentic AI System...")
    system = AgenticAISystem(config)
    
    # Setup knowledge base
    setup_native_ads_knowledge_base(system)
    
    # Example URLs untuk testing
    # NOTE: Ganti dengan URL artikel berita asli untuk testing
    
    print("="*80)
    print("DEMO: Analyzing Sample Articles")
    print("="*80 + "\n")
    
    # Contoh 1: Artikel Editorial (berita murni)
    print("\n[TEST 1] Editorial Content")
    print("-" * 80)
    editorial_url = input("Enter URL artikel editorial (atau Enter untuk skip): ").strip()
    if editorial_url:
        analyze_article(system, editorial_url, expected_label="Editorial Content")
    
    # Contoh 2: Native Advertising
    print("\n[TEST 2] Native Advertising")
    print("-" * 80)
    native_ads_url = input("Enter URL native ads (atau Enter untuk skip): ").strip()
    if native_ads_url:
        analyze_article(system, native_ads_url, expected_label="Native Advertising")
    
    # Contoh 3: Sponsored Content
    print("\n[TEST 3] Sponsored Content")
    print("-" * 80)
    sponsored_url = input("Enter URL sponsored content (atau Enter untuk skip): ").strip()
    if sponsored_url:
        analyze_article(system, sponsored_url, expected_label="Sponsored Content")
    
    # System Statistics
    print("\n" + "="*80)
    print("SYSTEM STATISTICS")
    print("="*80)
    
    stats = system.get_system_stats()
    print(f"\nKnowledge Base:")
    print(f"  Total Documents: {stats['retriever_stats']['total_documents']}")
    print(f"  Embedding Model: {stats['retriever_stats']['embedding_model']}")
    
    print(f"\nFeedback Summary:")
    feedback_summary = stats['feedback_summary']
    if feedback_summary.get('total_feedback', 0) > 0:
        print(f"  Total Feedback: {feedback_summary['total_feedback']}")
        print(f"  Avg Quality Score: {feedback_summary['avg_quality_score']:.2f}")
        print(f"  Needs Improvement: {feedback_summary['needs_improvement_count']}")
    else:
        print("  No feedback collected yet")
    
    # Export option
    print("\n" + "="*80)
    export = input("\nExport feedback data untuk training? (y/n): ").strip().lower()
    if export == 'y':
        output_file = 'data/native_ads_training_data.json'
        system.feedback_agent.export_training_data(output_file)
        print(f"✓ Training data exported to: {output_file}")
    
    print("\n" + "="*80)
    print("Demo selesai!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
