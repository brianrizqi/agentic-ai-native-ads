"""
Dataset Executor - Test Agentic AI dengan Dataset yang Sudah Dikonversi
Demonstrasi penggunaan dataset LLM untuk native ads detection
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import random
from main import AgenticAISystem


def load_converted_dataset(format_type='qna'):
    """
    Load dataset yang sudah dikonversi.
    
    Args:
        format_type: 'qna', 'instruction', atau 'chat'
    
    Returns:
        List of dataset samples
    """
    if format_type == 'qna':
        file_path = 'data/llm_dataset_qna.json'
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif format_type == 'instruction':
        file_path = 'data/llm_dataset_instruction.jsonl'
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    elif format_type == 'chat':
        file_path = 'data/llm_dataset_chat.jsonl'
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f"Unknown format: {format_type}")
    
    print(f"✓ Loaded {len(data)} samples from {file_path}")
    return data


def build_knowledge_base_from_dataset(dataset, num_examples=50):
    """
    Build knowledge base dari dataset untuk RAG.
    
    Args:
        dataset: Dataset samples
        num_examples: Jumlah contoh untuk knowledge base
    
    Returns:
        List of knowledge base documents
    """
    print(f"\nBuilding knowledge base from {num_examples} examples...")
    
    kb_docs = []
    
    # Ambil sample dari setiap kategori
    native_ads_samples = [s for s in dataset if 'native' in s.get('label', '').lower()]
    berita_samples = [s for s in dataset if 'berita' in s.get('label', '').lower()]
    
    # Sample dari masing-masing kategori
    native_examples = random.sample(native_ads_samples, min(num_examples//2, len(native_ads_samples)))
    berita_examples = random.sample(berita_samples, min(num_examples//2, len(berita_samples)))
    
    # Format sebagai knowledge base
    for sample in native_examples:
        context = sample.get('context', '')[:500]
        doc = f"[NATIVE ADS EXAMPLE] {context}"
        kb_docs.append(doc)
    
    for sample in berita_examples:
        context = sample.get('context', '')[:500]
        doc = f"[BERITA MURNI EXAMPLE] {context}"
        kb_docs.append(doc)
    
    print(f"✓ Created knowledge base with {len(kb_docs)} documents")
    print(f"  - Native ads examples: {len(native_examples)}")
    print(f"  - Berita murni examples: {len(berita_examples)}")
    
    return kb_docs


def test_with_random_samples(system, dataset, num_tests=5):
    """
    Test sistem dengan random samples dari dataset.
    
    Args:
        system: AgenticAISystem instance
        dataset: Dataset samples
        num_tests: Jumlah test samples
    """
    print("\n" + "="*80)
    print(f"TESTING WITH {num_tests} RANDOM SAMPLES")
    print("="*80)
    
    # Ambil random samples
    test_samples = random.sample(dataset, num_tests)
    
    results = []
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\n[TEST {i}/{num_tests}]")
        print("-" * 80)
        
        context = sample.get('context', '')
        true_label = sample.get('label', '')
        
        print(f"True Label: {true_label}")
        print(f"Content: {context[:200]}...")
        
        # Simulate classification (tanpa scraping, langsung dari dataset)
        # Kita akan gunakan preprocessing agent saja
        from agents.preprocessing_agent import PreprocessingAgent
        from agents.retriever_agent import RetrieverAgent
        from agents.llm_classifier_agent import LLMClassifierAgent
        from agents.explanation_agent import ExplanationAgent
        
        # Preprocess
        preprocessor = PreprocessingAgent()
        preprocessed = preprocessor.process({
            'url': 'dataset_sample',
            'title': f'Sample {i}',
            'text': context,
            'metadata': {}
        })
        
        # Retrieve context
        retrieved = system.retriever_agent.retrieve(preprocessed)
        
        # Classify (akan gunakan fallback jika tidak ada API key)
        classification = system.llm_classifier.classify(preprocessed, retrieved)
        
        # Explanation
        explanation = system.explanation_agent.explain(classification, preprocessed, retrieved)
        
        # Compare
        predicted_label = classification.get('label', '')
        confidence = classification.get('confidence', 0)
        
        is_correct = (
            ('native' in predicted_label.lower() and 'native' in true_label.lower()) or
            ('editorial' in predicted_label.lower() and 'berita' in true_label.lower()) or
            ('berita' in predicted_label.lower() and 'berita' in true_label.lower())
        )
        
        print(f"\nPredicted: {predicted_label} (confidence: {confidence:.2%})")
        print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        print(f"\nExplanation: {explanation.get('summary', 'N/A')[:150]}...")
        
        results.append({
            'true_label': true_label,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'correct': is_correct
        })
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    accuracy = sum(1 for r in results if r['correct']) / len(results)
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    
    print(f"Accuracy: {accuracy:.2%} ({sum(1 for r in results if r['correct'])}/{len(results)})")
    print(f"Average Confidence: {avg_confidence:.2%}")
    
    return results


def main():
    """Main execution function."""
    
    print("="*80)
    print("DATASET EXECUTOR - AGENTIC AI NATIVE ADS DETECTION")
    print("Testing dengan Dataset yang Sudah Dikonversi")
    print("="*80 + "\n")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize system
    print("Initializing Agentic AI System...")
    system = AgenticAISystem(config)
    
    # Load converted dataset
    print("\nLoading converted dataset...")
    dataset = load_converted_dataset(format_type='qna')
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    
    # Count labels
    labels = {}
    for sample in dataset:
        label = sample.get('label', 'unknown')
        labels[label] = labels.get(label, 0) + 1
    
    print(f"  Label distribution:")
    for label, count in labels.items():
        print(f"    - {label}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Build knowledge base dari dataset
    kb_docs = build_knowledge_base_from_dataset(dataset, num_examples=100)
    system.add_knowledge_base_documents(kb_docs)
    
    # Test dengan random samples
    print("\n" + "="*80)
    print("DEMO: Testing Agentic AI dengan Dataset Samples")
    print("="*80)
    
    num_tests = int(input("\nBerapa sample yang ingin di-test? (default: 5): ").strip() or "5")
    
    results = test_with_random_samples(system, dataset, num_tests=num_tests)
    
    # Export results
    output_file = 'data/test_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Test results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Fine-tune LLM dengan dataset instruction format
2. Evaluate dengan test set yang lebih besar
3. Improve knowledge base dengan lebih banyak examples
4. Deploy sistem untuk production testing
    """)


if __name__ == "__main__":
    main()
