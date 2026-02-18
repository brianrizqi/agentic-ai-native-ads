"""
Evaluation Script - BERTScore & LLM-as-a-Judge
Evaluates the Agentic AI system performance on native ads detection
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import argparse
from tqdm import tqdm
import numpy as np

# Import agents
import sys
sys.path.append(str(Path(__file__).parent))
from agents.web_agent import WebAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.retriever_agent import RetrieverAgent
from agents.llm_classifier_agent import LLMClassifierAgent
from agents.explanation_agent import ExplanationAgent


class EvaluationFramework:
    """
    Comprehensive evaluation framework using:
    1. BERTScore - for semantic similarity
    2. LLM-as-a-Judge - for classification quality
    3. Traditional metrics - accuracy, precision, recall, F1
    """
    
    def __init__(self, model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                 judge_model: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                 lora_path: str = None):
        """
        Initialize evaluation framework.
        
        Args:
            model_name: Model to evaluate
            judge_model: Model to use as judge
            lora_path: Path to LoRA adapters (optional)
        """
        self.model_name = model_name
        self.judge_model = judge_model
        
        # Initialize BERTScore
        try:
            from bert_score import BERTScorer
            self.bert_scorer = BERTScorer(lang="id", rescale_with_baseline=True)
            print("[OK] BERTScore initialized")
        except ImportError:
            print("[WARNING] bert-score not installed. Run: pip install bert-score")
            self.bert_scorer = None
        
        # Initialize agents
        print(f"[INFO] Initializing agents with model: {model_name}")
        self.preprocess_agent = PreprocessingAgent()
        
        dataset_path = 'data/llm_dataset_qna.json'
        if Path(dataset_path).exists():
            self.retriever_agent = RetrieverAgent(
                vector_db_path=dataset_path,
                embedding_model='sentence-transformers/all-MiniLM-L6-v2'
            )
        else:
            print(f"[WARNING] Dataset not found: {dataset_path}")
            self.retriever_agent = None
        
        self.classifier_agent = LLMClassifierAgent(
            provider='huggingface',
            model_name=model_name,
            lora_path=lora_path
        )
        
        # Initialize judge (separate instance)
        print(f"[INFO] Initializing LLM Judge: {judge_model}")
        self.judge_agent = LLMClassifierAgent(
            provider='huggingface',
            model_name=judge_model
        )
    
    def evaluate_sample(self, content: str, ground_truth_label: str) -> Dict[str, Any]:
        """
        Evaluate a single sample.
        
        Args:
            content: Article content
            ground_truth_label: True label
            
        Returns:
            Evaluation results
        """
        # Preprocess
        scraped_data = {
            'text': content,
            'title': '',
            'paragraphs': [content],
            'metadata': {}
        }
        preprocessed_data = self.preprocess_agent.process(scraped_data)
        
        # Retrieve context
        if self.retriever_agent:
            context = self.retriever_agent.retrieve(preprocessed_data)
        else:
            context = []
        
        # Classify
        classification = self.classifier_agent.classify(preprocessed_data, context)
        
        # Evaluate
        results = {
            'predicted_label': classification.get('label', 'unknown'),
            'confidence': classification.get('confidence', 0.0),
            'reasoning': classification.get('reasoning', ''),
            'ground_truth': ground_truth_label,
            'correct': self._is_correct(classification.get('label'), ground_truth_label),
            'preprocessed_data': preprocessed_data,
            'classification': classification
        }
        
        return results
    
    def _is_correct(self, predicted: str, ground_truth: str) -> bool:
        """Check if prediction is correct."""
        # Normalize labels
        pred_norm = predicted.lower().replace(' ', '_').replace('-', '_')
        gt_norm = ground_truth.lower().replace(' ', '_').replace('-', '_')
        
        # Map variations
        native_ads_variants = ['native_ads', 'native_advertising', 'advertorial']
        editorial_variants = ['editorial', 'editorial_content', 'news']
        
        if pred_norm in native_ads_variants and gt_norm in native_ads_variants:
            return True
        if pred_norm in editorial_variants and gt_norm in editorial_variants:
            return True
        
        return pred_norm == gt_norm
    
    def compute_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute BERTScore for predictions vs references.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            BERTScore metrics (P, R, F1)
        """
        if not self.bert_scorer:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        P, R, F1 = self.bert_scorer.score(predictions, references)
        
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
    
    def llm_as_judge(self, prediction: Dict[str, Any], ground_truth: str) -> Dict[str, Any]:
        """
        Use LLM as a judge to evaluate classification quality.
        
        Args:
            prediction: Prediction dict with label, confidence, reasoning
            ground_truth: Ground truth label
            
        Returns:
            Judge evaluation
        """
        prompt = f"""[INST] You are an expert evaluator for native advertising detection systems.

Evaluate the following classification result:

PREDICTED LABEL: {prediction.get('label', 'unknown')}
CONFIDENCE: {prediction.get('confidence', 0.0):.2%}
REASONING: {prediction.get('reasoning', 'N/A')}

GROUND TRUTH LABEL: {ground_truth}

Evaluate on these criteria (score 1-5 for each):
1. **Correctness**: Is the predicted label correct?
2. **Confidence Calibration**: Is the confidence level appropriate?
3. **Reasoning Quality**: Is the reasoning clear and logical?
4. **Overall Quality**: Overall assessment of the classification

Provide output in JSON format:
{{
  "correctness_score": 1-5,
  "confidence_calibration_score": 1-5,
  "reasoning_quality_score": 1-5,
  "overall_quality_score": 1-5,
  "judge_comment": "brief comment"
}}
[/INST]"""
        
        try:
            # Use judge model
            response = self.judge_agent.client(
                prompt,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                truncation=True
            )
            response_text = response[0]['generated_text']
            
            # Strip prompt
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt):]
            
            # Parse JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response_text[start:end]
                judge_result = json.loads(json_str)
            else:
                raise ValueError("No JSON found")
            
            return judge_result
            
        except Exception as e:
            print(f"[WARNING] LLM Judge failed: {e}")
            return {
                'correctness_score': 0,
                'confidence_calibration_score': 0,
                'reasoning_quality_score': 0,
                'overall_quality_score': 0,
                'judge_comment': 'Evaluation failed'
            }
    
    def evaluate_dataset(self, dataset_path: str, num_samples: int = 100) -> Dict[str, Any]:
        """
        Evaluate on dataset.
        
        Args:
            dataset_path: Path to dataset JSON
            num_samples: Number of samples to evaluate
            
        Returns:
            Comprehensive evaluation results
        """
        print(f"\n{'='*80}")
        print(f"EVALUATION - {self.model_name}")
        print(f"{'='*80}\n")
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Sample
        if len(dataset) > num_samples:
            import random
            dataset = random.sample(dataset, num_samples)
        
        print(f"Evaluating on {len(dataset)} samples...\n")
        
        # Evaluate each sample
        results = []
        predictions_text = []
        references_text = []
        judge_scores = []
        
        for sample in tqdm(dataset, desc="Evaluating"):
            # Get content and label
            if 'input' in sample:
                content = sample['input']
                label = sample.get('output', '').split('\n')[0].replace('**Klasifikasi**: ', '')
            elif 'context' in sample:
                content = sample['context']
                label = sample.get('label', 'unknown')
            else:
                continue
            
            # Evaluate
            result = self.evaluate_sample(content, label)
            results.append(result)
            
            # For BERTScore
            predictions_text.append(result['reasoning'])
            references_text.append(f"This is {label}")
            
            # LLM as Judge
            judge_result = self.llm_as_judge(result['classification'], label)
            judge_scores.append(judge_result)
        
        # Compute metrics
        accuracy = sum(r['correct'] for r in results) / len(results)
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        # BERTScore
        bert_scores = self.compute_bertscore(predictions_text, references_text)
        
        # LLM Judge scores
        avg_judge_scores = {
            'correctness': np.mean([s.get('correctness_score', 0) for s in judge_scores]),
            'confidence_calibration': np.mean([s.get('confidence_calibration_score', 0) for s in judge_scores]),
            'reasoning_quality': np.mean([s.get('reasoning_quality_score', 0) for s in judge_scores]),
            'overall_quality': np.mean([s.get('overall_quality_score', 0) for s in judge_scores])
        }
        
        # Compile results
        evaluation_results = {
            'model': self.model_name,
            'num_samples': len(results),
            'traditional_metrics': {
                'accuracy': accuracy,
                'avg_confidence': avg_confidence
            },
            'bertscore': bert_scores,
            'llm_judge_scores': avg_judge_scores,
            'detailed_results': results,
            'judge_evaluations': judge_scores
        }
        
        # Print summary
        self._print_summary(evaluation_results)
        
        return evaluation_results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"Model: {results['model']}")
        print(f"Samples Evaluated: {results['num_samples']}\n")
        
        print("--- Traditional Metrics ---")
        print(f"Accuracy: {results['traditional_metrics']['accuracy']:.2%}")
        print(f"Average Confidence: {results['traditional_metrics']['avg_confidence']:.2%}\n")
        
        print("--- BERTScore ---")
        print(f"Precision: {results['bertscore']['precision']:.4f}")
        print(f"Recall: {results['bertscore']['recall']:.4f}")
        print(f"F1: {results['bertscore']['f1']:.4f}\n")
        
        print("--- LLM-as-a-Judge Scores (1-5 scale) ---")
        for metric, score in results['llm_judge_scores'].items():
            print(f"{metric.replace('_', ' ').title()}: {score:.2f}/5.0")
        
        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Agentic AI System')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2',
                       help='Model to evaluate')
    parser.add_argument('--judge-model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2',
                       help='Model to use as judge')
    parser.add_argument('--lora-path', type=str, default=None,
                       help='Path to LoRA adapters (optional)')
    parser.add_argument('--dataset', type=str, default='data/llm_dataset_instruction.json',
                       help='Dataset path')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to evaluate')
    parser.add_argument('--output', type=str, default='data/evaluation_results.json',
                       help='Output file for results')
    args = parser.parse_args()
    
    # Check dataset
    if not Path(args.dataset).exists():
        print(f"[ERROR] Dataset not found: {args.dataset}")
        print("Run: python tools/dataset_converter.py first")
        return
    
    # Initialize evaluator
    evaluator = EvaluationFramework(
        model_name=args.model,
        judge_model=args.judge_model,
        lora_path=args.lora_path
    )
    
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
    
    print(f"[OK] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
