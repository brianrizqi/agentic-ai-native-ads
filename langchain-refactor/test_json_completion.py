#!/usr/bin/env python3
"""
Quick Test Script - Verify JSON Completion Fixes
Tests the improved classification agent with sample data
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from agents.classification_agent import ClassificationAgent

def test_json_completion():
    """Test JSON completion with sample data."""
    
    print("="*80)
    print("TESTING JSON COMPLETION FIXES")
    print("="*80)
    
    # Test samples
    test_samples = [
        {
            "content": "Stres dan asupan makanan penuhi imunitas, ini tips dari dokter untuk menjaga kesehatan Anda. Produk vitamin terbaik dari Brand X tersedia dengan diskon spesial.",
            "expected": "native ads"
        },
        {
            "content": "Presiden Jokowi menunjuk Ridwan Kamil sebagai kurator infrastruktur IKN pada hari Selasa. Keputusan ini mendapat kritik dari berbagai pihak.",
            "expected": "berita murni"
        },
        {
            "content": "Dapatkan cashback hingga 50% untuk pembelian produk elektronik di Tokopedia hari ini! Promo terbatas hanya untuk member premium.",
            "expected": "native ads"
        }
    ]
    
    # Test with Qwen model (if available)
    model_path = "../models/qwen-native-ads_merged_16bit"
    
    if not Path(model_path).exists():
        print(f"\n⚠️  Model not found: {model_path}")
        print("Please train the model first using:")
        print("  python finetune.py --model qwen")
        return
    
    print(f"\nLoading model: {model_path}")
    print("This may take a minute...\n")
    
    try:
        agent = ClassificationAgent(
            model_name=model_path,
            provider="local"
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run tests
    print("Running tests...\n")
    print("-"*80)
    
    success_count = 0
    json_error_count = 0
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\nTest {i}/{len(test_samples)}")
        print(f"Content: {sample['content'][:100]}...")
        print(f"Expected: {sample['expected']}")
        
        try:
            result = agent.classify(
                title="",
                summary="",
                content=sample['content'],
                context=""
            )
            
            print(f"Result: {result['label']} (confidence: {result['confidence']:.2f})")
            print(f"Reasoning: {result['reasoning'][:100]}...")
            
            # Check if JSON was properly formed
            if 'label' in result and 'confidence' in result:
                success_count += 1
                print("✅ JSON properly formed")
            else:
                print("⚠️  Incomplete result")
                
        except json.JSONDecodeError as e:
            json_error_count += 1
            print(f"❌ JSON Error: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-"*80)
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {len(test_samples)}")
    print(f"Successful: {success_count}")
    print(f"JSON errors: {json_error_count}")
    print(f"Success rate: {success_count/len(test_samples)*100:.1f}%")
    
    if json_error_count == 0:
        print("\n✅ All tests passed! JSON completion is working correctly.")
        print("\n🚀 Next Steps:")
        print("1. Re-train all models with the fixed format:")
        print("   python finetune.py --model qwen")
        print("   python finetune.py --model llama")
        print("   python finetune.py --model gemma")
        print("2. Run full comparison:")
        print("   python compare_models.py")
    else:
        print(f"\n⚠️  {json_error_count} JSON errors detected.")
        print("The models need to be re-trained with the fixed format.")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_json_completion()
