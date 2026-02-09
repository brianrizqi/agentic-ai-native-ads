"""
Example usage of Instructor Classification Agent with Gemma v7
Demonstrates how to use the instructor-based agent for native ads detection
"""

import logging
from agents.instructor_classification_agent import InstructorClassificationAgent, ClassificationResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_single_classification():
    """Example: Classify a single article."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Article Classification")
    print("="*80 + "\n")
    
    # Initialize agent
    agent = InstructorClassificationAgent(
        model_path="../models/gemma-native-ads-v7_merged_16bit",
        temperature=0.1,
        use_instructor=True  # Enable instructor for structured output
    )
    
    # Print model info
    info = agent.get_model_info()
    print(f"Model loaded: {info['model_path']}")
    print(f"Device: {info['device']}")
    print(f"Parameters: {info['model_params']:.2f}B")
    print(f"Instructor enabled: {info['use_instructor']}\n")
    
    # Example article (Native Ads)
    title = "Promo Spesial BRI di HUT ke-128, Diskon Hingga Rp1.28 Juta"
    content = """
    Bank Rakyat Indonesia (BRI) merayakan HUT ke-128 dengan memberikan promo spesial 
    untuk nasabah setia. Dapatkan diskon hingga Rp1.28 juta untuk berbagai produk 
    perbankan termasuk KPR, kredit kendaraan, dan kartu kredit. Promo berlaku hingga 
    akhir bulan ini. Jangan lewatkan kesempatan emas ini untuk mendapatkan layanan 
    terbaik dari BRI dengan harga spesial!
    """
    
    # Classify
    result = agent.classify(title=title, content=content)
    
    # Display results
    print("CLASSIFICATION RESULT:")
    print(f"  Label: {result.label}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Reasoning: {result.reasoning}")
    print()


def example_batch_classification():
    """Example: Classify multiple articles in batch."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Classification")
    print("="*80 + "\n")
    
    # Initialize agent
    agent = InstructorClassificationAgent(
        model_path="../models/gemma-native-ads-v7_merged_16bit",
        temperature=0.1,
        use_instructor=True
    )
    
    # Multiple articles
    articles = [
        {
            "title": "Presiden Jokowi Tunjuk Ridwan Kamil sebagai Kurator Infrastruktur IKN",
            "content": "Presiden Joko Widodo resmi menunjuk Ridwan Kamil sebagai kurator infrastruktur Ibu Kota Nusantara (IKN) pada hari Selasa. Keputusan ini diambil setelah melalui berbagai pertimbangan dan konsultasi dengan berbagai pihak."
        },
        {
            "title": "Dapatkan Cashback 50% untuk Pembelian Produk Elektronik di Tokopedia",
            "content": "Tokopedia menghadirkan promo spesial cashback hingga 50% untuk pembelian produk elektronik. Promo berlaku untuk berbagai kategori mulai dari smartphone, laptop, hingga peralatan rumah tangga. Buruan belanja sekarang!"
        },
        {
            "title": "KPK Tetapkan Tersangka Baru dalam Kasus Korupsi Bansos",
            "content": "Komisi Pemberantasan Korupsi (KPK) menetapkan tersangka baru dalam kasus korupsi bantuan sosial. Tersangka berinisial AS diduga menerima suap senilai Rp2 miliar. KPK akan segera melakukan penyidikan lebih lanjut."
        }
    ]
    
    # Classify batch
    results = agent.classify_batch(articles, show_progress=True)
    
    # Display results
    print("\nBATCH CLASSIFICATION RESULTS:")
    print("-" * 80)
    for i, (article, result) in enumerate(zip(articles, results), 1):
        print(f"\n{i}. {article['title'][:60]}...")
        print(f"   Label: {result.label}")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Reasoning: {result.reasoning}")
    print()


def example_comparison_with_without_instructor():
    """Example: Compare results with and without instructor."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Comparison - With vs Without Instructor")
    print("="*80 + "\n")
    
    title = "Vivo Perluas Jangkauan Layanan Purnajual di Indonesia"
    content = """
    Vivo Indonesia terus memperkuat komitmennya dalam memberikan layanan terbaik 
    kepada konsumen dengan memperluas jangkauan layanan purnajual. Kini, Vivo 
    Service Center tersedia di lebih dari 100 kota di seluruh Indonesia. Dengan 
    teknisi bersertifikat dan suku cadang original, Vivo memastikan smartphone 
    Anda selalu dalam kondisi prima.
    """
    
    # With Instructor
    print("1. WITH INSTRUCTOR (Structured Output):")
    agent_with = InstructorClassificationAgent(
        model_path="../models/gemma-native-ads-v7_merged_16bit",
        use_instructor=True
    )
    result_with = agent_with.classify(title=title, content=content)
    print(f"   Label: {result_with.label}")
    print(f"   Confidence: {result_with.confidence:.2%}")
    print(f"   Reasoning: {result_with.reasoning}\n")
    
    # Without Instructor
    print("2. WITHOUT INSTRUCTOR (Standard Generation):")
    agent_without = InstructorClassificationAgent(
        model_path="../models/gemma-native-ads-v7_merged_16bit",
        use_instructor=False
    )
    result_without = agent_without.classify(title=title, content=content)
    print(f"   Label: {result_without.label}")
    print(f"   Confidence: {result_without.confidence:.2%}")
    print(f"   Reasoning: {result_without.reasoning}\n")
    
    print("Note: Instructor mode provides more reliable structured output")
    print("      and better error handling compared to standard generation.\n")


def example_error_handling():
    """Example: Demonstrate error handling."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Error Handling")
    print("="*80 + "\n")
    
    agent = InstructorClassificationAgent(
        model_path="../models/gemma-native-ads-v7_merged_16bit",
        use_instructor=True
    )
    
    # Test with empty content
    print("Test 1: Empty content")
    result = agent.classify(title="", content="")
    print(f"  Result: {result.label} (confidence: {result.confidence:.2%})")
    print(f"  Reasoning: {result.reasoning}\n")
    
    # Test with very long content
    print("Test 2: Very long content (will be truncated)")
    long_content = "Lorem ipsum dolor sit amet. " * 100
    result = agent.classify(title="Test Article", content=long_content)
    print(f"  Result: {result.label} (confidence: {result.confidence:.2%})")
    print(f"  Reasoning: {result.reasoning}\n")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("INSTRUCTOR CLASSIFICATION AGENT - EXAMPLES")
    print("Gemma v7 Model (90.50% Accuracy)")
    print("="*80)
    
    try:
        # Run examples
        example_single_classification()
        example_batch_classification()
        example_comparison_with_without_instructor()
        example_error_handling()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Model exists at: ../models/gemma-native-ads-v7_merged_16bit")
        print("  2. All dependencies are installed (see requirements-instructor.txt)")
        print("  3. You have sufficient GPU/CPU memory\n")


if __name__ == "__main__":
    main()
