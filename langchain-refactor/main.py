"""
Main Entry Point for LangChain Refactored System
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import json

from chains.full_pipeline_chain import FullPipelineChain

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_results(url: str, result: dict):
    """Save classification results to results/ folder with timestamp."""
    
    # Create results directory
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract domain from URL for filename
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.replace('www.', '').replace('.', '_')
    
    # Prepare data
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "url": url,
        "title": result.get('title', 'N/A'),
        "classification": result.get('classification', {}),
        "explanation": result.get('explanation', 'N/A'),
        "metadata": result.get('metadata', {})
    }
    
    # Save as JSON
    json_file = results_dir / f"{timestamp}_{domain}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    # Save as TXT (human-readable)
    txt_file = results_dir / f"{timestamp}_{domain}.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("NATIVE ADS CLASSIFICATION RESULT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {save_data['timestamp']}\n")
        f.write(f"URL: {url}\n")
        f.write(f"Title: {save_data['title']}\n\n")
        
        classification = save_data['classification']
        f.write("CLASSIFICATION:\n")
        f.write(f"  Label: {classification.get('label', 'N/A')}\n")
        f.write(f"  Confidence: {classification.get('confidence', 0):.2f}\n")
        f.write(f"  Reasoning: {classification.get('reasoning', 'N/A')}\n\n")
        
        if save_data.get('explanation'):
            f.write("EXPLANATION:\n")
            f.write(f"{save_data['explanation']}\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"\n💾 Results saved:")
    print(f"   JSON: {json_file}")
    print(f"   TXT:  {txt_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Native Ads Detection System (LangChain Refactored)'
    )
    parser.add_argument('--url', type=str, help='URL to analyze')
    parser.add_argument('--urls-file', type=str, help='File containing URLs (one per line)')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                       help='LLM model name')
    parser.add_argument('--provider', type=str, default='openai',
                       choices=['openai', 'openrouter', 'huggingface', 'local'],
                       help='LLM provider')
    parser.add_argument('--api-key', type=str, help='API key for LLM provider')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--collect', type=int, help='Collect and build dataset with N articles')
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
        
        # Save results automatically
        if result.get('status') == 'success':
            save_results(args.url, result)
        
    elif args.urls_file:
        # Multiple URLs from file
        logger.info(f"Processing URLs from file: {args.urls_file}")
        with open(args.urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        results = pipeline.run_batch(urls)
    
    elif args.collect:
        # Automated dataset collection
        logger.info(f"Triggering automated collection for {args.collect} articles...")
        from dataset_builder import DatasetBuilder
        builder = DatasetBuilder(
            model_name=args.model,
            provider=args.provider,
            api_key=args.api_key
        )
        # Manual high-value native ads URLs provided by user
        MANUAL_URLS = [
            "https://tekno.sindonews.com/read/474336/776/zte-dan-telkomsel-hadirkan-skenario-penggunaan-5g-pada-peluncuran-telkomsel-5g-1625450849",
            "https://tekno.sindonews.com/read/464846/776/perkuat-pasar-di-indonesia-vivo-perluas-jangkauan-layanan-purnajual-1624500405",
            "https://tekno.sindonews.com/read/462176/776/rekomendasi-smartwatch-dengan-harga-di-bawah-rp1-juta-di-shopee-1624266370",
            "https://tekno.sindonews.com/read/447848/776/inovasi-hp-kamera-terbaik-vivo-pada-7-tahun-kiprahnya-di-indonesia-1622988418",
            "https://otomotif.sindonews.com/read/787347/778/begini-sensasi-mengendarai-all-new-hondabr-v-untuk-perjalanan-jauh-bersama-keluarga-1654236414",
            "https://otomotif.sindonews.com/read/767061/778/service-improvement-mitsubishi-dalam-gelaran-iims-2022-1652324702",
            "https://otomotif.sindonews.com/read/747837/778/perbandingan-antara-honda-br-v-lama-dan-baru-signifikan-bedanya-1650362675",
            "https://otomotif.sindonews.com/read/740347/778/5-keunggulan-all-new-honda-br-v-nyaman-untuk-harian-hingga-mudik-lebaran-1649682288",
            "https://otomotif.sindonews.com/read/737447/778/sediakan-layanan-after-sales-mitsubishi-tawarkan-banyak-diskon-di-iims-2022-1649408664",
            "https://otomotif.sindonews.com/read/737143/778/hadir-di-iims-2022-mitsubishi-motors-berikan-booth-experience-hingga-promo-menarik-1649394271"
        ]
        output_paths = builder.collect_and_build(
            target_count=args.collect,
            manual_urls=MANUAL_URLS
        )
        print(f"\n✅ Dataset collection complete!")
        for fmt, path in output_paths.items():
            print(f"   - {fmt.upper()}: {path}")
        return
    
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
