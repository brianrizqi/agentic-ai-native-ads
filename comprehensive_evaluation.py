"""
Comprehensive Evaluation Script dengan Confusion Matrix
Untuk publikasi penelitian postdoc
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import agents
import sys
sys.path.append(str(Path(__file__).parent))
from agents.preprocessing_agent import PreprocessingAgent
from agents.retriever_agent import RetrieverAgent
from agents.llm_classifier_agent import LLMClassifierAgent


class ComprehensiveEvaluator:
    """Evaluasi komprehensif untuk paper penelitian."""
    
    def __init__(self, model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2'):
        self.model_name = model_name
        
        # Initialize agents
        self.preprocess_agent = PreprocessingAgent()
        
        dataset_path = 'data/llm_dataset_qna.json'
        if Path(dataset_path).exists():
            self.retriever_agent = RetrieverAgent(
                vector_db_path=dataset_path,
                embedding_model='sentence-transformers/all-MiniLM-L6-v2'
            )
        else:
            self.retriever_agent = None
        
        self.classifier_agent = LLMClassifierAgent(
            provider='huggingface',
            model_name=model_name
        )
    
    def evaluate_sample(self, content: str, ground_truth: str) -> Dict[str, Any]:
        """Evaluate single sample."""
        # Preprocess
        scraped_data = {'text': content, 'title': '', 'paragraphs': [content]}
        preprocessed_data = self.preprocess_agent.process(scraped_data)
        
        # Retrieve context
        context = []
        if self.retriever_agent:
            context = self.retriever_agent.retrieve(preprocessed_data)
        
        # Classify
        classification = self.classifier_agent.classify(preprocessed_data, context)
        
        return {
            'predicted': classification.get('label', 'unknown'),
            'ground_truth': ground_truth,
            'confidence': classification.get('confidence', 0.0),
            'reasoning': classification.get('reasoning', '')
        }
    
    def evaluate_dataset(self, dataset_path: str, num_samples: int = 500):
        """Evaluate on dataset with comprehensive metrics."""
        
        print("="*80)
        print("COMPREHENSIVE EVALUATION - Native Ads Detection")
        print("="*80)
        print(f"Model: {self.model_name}")
        print(f"Samples: {num_samples}")
        print("="*80 + "\n")
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Sample
        if len(dataset) > num_samples:
            import random
            random.seed(42)
            dataset = random.sample(dataset, num_samples)
        
        # Evaluate
        results = []
        y_true = []
        y_pred = []
        confidences = []
        
        for sample in tqdm(dataset, desc="Evaluating"):
            # Get content and label
            if 'input' in sample:
                content = sample['input']
                # Extract label from output
                output = sample.get('output', '')
                if 'native ads' in output.lower():
                    label = 'native ads'
                else:
                    label = 'berita murni'
            elif 'context' in sample:
                content = sample['context']
                label = sample.get('label', 'berita murni')
            else:
                continue
            
            # Normalize label
            if 'native' in label.lower():
                label = 'native ads'
            else:
                label = 'berita murni'
            
            # Evaluate
            result = self.evaluate_sample(content, label)
            results.append(result)
            
            y_true.append(result['ground_truth'])
            y_pred.append(result['predicted'])
            confidences.append(result['confidence'])
        
        # Compute metrics
        metrics = self._compute_metrics(y_true, y_pred, confidences)
        
        # Generate confusion matrix
        self._plot_confusion_matrix(y_true, y_pred, save_path='data/confusion_matrix.png')
        
        # Save results
        output = {
            'model': self.model_name,
            'num_samples': len(results),
            'metrics': metrics,
            'detailed_results': results
        }
        
        # Print summary
        self._print_summary(metrics)
        
        return output
    
    def _compute_metrics(self, y_true: List[str], y_pred: List[str], 
                        confidences: List[float]) -> Dict[str, Any]:
        """Compute all metrics."""
        
        # Traditional metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['native ads', 'berita murni'])
        
        # Per-class metrics
        native_ads_metrics = report.get('native ads', {})
        berita_murni_metrics = report.get('berita murni', {})
        
        return {
            'accuracy': accuracy,
            'macro_avg': {
                'precision': report['macro avg']['precision'],
                'recall': report['macro avg']['recall'],
                'f1_score': report['macro avg']['f1-score']
            },
            'native_ads': {
                'precision': native_ads_metrics.get('precision', 0),
                'recall': native_ads_metrics.get('recall', 0),
                'f1_score': native_ads_metrics.get('f1-score', 0),
                'support': native_ads_metrics.get('support', 0)
            },
            'berita_murni': {
                'precision': berita_murni_metrics.get('precision', 0),
                'recall': berita_murni_metrics.get('recall', 0),
                'f1_score': berita_murni_metrics.get('f1-score', 0),
                'support': berita_murni_metrics.get('support', 0)
            },
            'confusion_matrix': cm.tolist(),
            'avg_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences)
        }
    
    def _plot_confusion_matrix(self, y_true: List[str], y_pred: List[str], 
                               save_path: str = 'confusion_matrix.png'):
        """Plot and save confusion matrix."""
        
        cm = confusion_matrix(y_true, y_pred, labels=['native ads', 'berita murni'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['native ads', 'berita murni'],
                   yticklabels=['native ads', 'berita murni'])
        plt.title('Confusion Matrix - Native Ads Detection')
        plt.ylabel('Ground Truth')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix saved to: {save_path}")
    
    def _print_summary(self, metrics: Dict[str, Any]):
        """Print evaluation summary."""
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80 + "\n")
        
        print("--- Overall Metrics ---")
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Macro Precision: {metrics['macro_avg']['precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_avg']['recall']:.4f}")
        print(f"Macro F1-Score: {metrics['macro_avg']['f1_score']:.4f}")
        print(f"Avg Confidence: {metrics['avg_confidence']:.4f} ± {metrics['confidence_std']:.4f}\n")
        
        print("--- Per-Class Metrics ---")
        print("\nNative Ads:")
        print(f"  Precision: {metrics['native_ads']['precision']:.4f}")
        print(f"  Recall: {metrics['native_ads']['recall']:.4f}")
        print(f"  F1-Score: {metrics['native_ads']['f1_score']:.4f}")
        print(f"  Support: {metrics['native_ads']['support']}")
        
        print("\nBerita Murni:")
        print(f"  Precision: {metrics['berita_murni']['precision']:.4f}")
        print(f"  Recall: {metrics['berita_murni']['recall']:.4f}")
        print(f"  F1-Score: {metrics['berita_murni']['f1_score']:.4f}")
        print(f"  Support: {metrics['berita_murni']['support']}")
        
        print("\n--- Confusion Matrix ---")
        cm = np.array(metrics['confusion_matrix'])
        print(f"\n                Predicted")
        print(f"                native ads  berita murni")
        print(f"Actual")
        print(f"native ads      {cm[0][0]:<11} {cm[0][1]}")
        print(f"berita murni    {cm[1][0]:<11} {cm[1][1]}")
        
        print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Evaluation')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--dataset', type=str, default='data/llm_dataset_instruction.json')
    parser.add_argument('--num-samples', type=int, default=500)
    parser.add_argument('--output', type=str, default='data/comprehensive_evaluation.json')
    args = parser.parse_args()
    
    # Check dataset
    if not Path(args.dataset).exists():
        print(f"[ERROR] Dataset not found: {args.dataset}")
        return
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(model_name=args.model)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        dataset_path=args.dataset,
        num_samples=args.num_samples
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to: {output_path}")
    print(f"✅ Confusion matrix saved to: data/confusion_matrix.png")


if __name__ == "__main__":
    main()
