"""
RAG Ablation Study: Impact of Context on Native Ads Detection
=============================================================
Membandingkan 3 skenario RAG:
1.  **Tanpa RAG (Baseline)**: Model hanya mengandalkan knowledge internal.
2.  **RAG FAISS (Internal)**: Menggunakan data asli dari `data/vectorstore/`.
3.  **RAG Eksternal (Irrelevant)**: Menggunakan data irrelevant dari `data/vectorstore_irrelevant/`.

Tujuan:
- Mengukur seberapa besar RAG meningkatkan akurasi klasifikasi.
- Mengamati dampak 'distractor context' (RAG eksternal) terhadap kualitas reasoning model.
"""

import os
import json
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Skip actual computation of BERTScore if libraries missing, same as evaluate_model.py
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# We'll import these inside classes to let the script be portable
# even if some deps are missing on the current machine.

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

RAG_MODES = {
    "none": {
        "name": "No RAG (Baseline)",
        "db_path": None,
        "description": "Classification without any external context."
    },
    "internal": {
        "name": "RAG FAISS (Internal Data)",
        "db_path": "data/vectorstore",
        "description": "RAG using relevant native ads training data."
    },
    "external": {
        "name": "RAG Irrelevant (External Data)",
        "db_path": "data/vectorstore_irrelevant",
        "description": "RAG using completely unrelated data (distractor context)."
    }
}

# ─── Implementation ───────────────────────────────────────────────────────────

class RAGAblationEvaluator:
    def __init__(self, model_name: str, provider: str, api_key: str, embedding_model: str):
        self.model_name = model_name
        self.provider = provider
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.vectorstores = {}
        
    def setup_rag_mode(self, mode: str):
        """Lazy load vectorstores."""
        if mode == "none":
            return None
        
        if mode not in self.vectorstores:
            db_path = RAG_MODES[mode]["db_path"]
            if not Path(db_path).exists():
                logger.error(f"Vectorstore not found at {db_path} for mode {mode}")
                return None
            
            try:
                from vector_stores.native_ads_vectorstore import NativeAdsVectorStore
                logger.info(f"Loading {mode} vectorstore from {db_path}...")
                self.vectorstores[mode] = NativeAdsVectorStore(
                    embedding_model=self.embedding_model,
                    persist_directory=db_path
                )
            except Exception as e:
                logger.error(f"Failed to load vectorstore for {mode}: {e}")
                return None
        
        return self.vectorstores[mode]

    def evaluate(self, test_data: List[Dict], num_samples: int, output_dir: Path):
        results = {}
        
        # Limit samples
        samples = test_data[:num_samples]
        
        for mode in RAG_MODES.keys():
            logger.info(f"\n🚀 Evaluating mode: {RAG_MODES[mode]['name']}")
            mode_results = self._run_inference_for_mode(mode, samples)
            results[mode] = mode_results
            
            # Save intermediate results
            mode_file = output_dir / f"results_{mode}.json"
            with open(mode_file, 'w', encoding='utf-8') as f:
                json.dump(mode_results, f, ensure_ascii=False, indent=2)
        
        return results

    def _run_inference_for_mode(self, mode: str, samples: List[Dict]) -> Dict:
        """Run classification pipeline for all samples in a specific RAG mode."""
        from agents.classification_agent import ClassificationAgent
        from agents.retrieval_agent import RetrievalAgent
        
        vstore = self.setup_rag_mode(mode)
        retriever = RetrievalAgent(vectorstore=vstore) if vstore else None
        
        # Initialize agent
        # Note: we use our improved ClassificationAgent from agents/classification_agent.py
        agent = ClassificationAgent(
            model_name=self.model_name,
            provider=self.provider,
            api_key=self.api_key,
            temperature=0.1 # Low temp for consistency in evaluation
        )
        
        predictions = []
        correct_count = 0
        total_time = 0
        
        for i, item in enumerate(samples):
            logger.info(f"   [{mode}] Processing sample {i+1}/{len(samples)}...")
            
            content = item.get('input', '')
            title = item.get('title', '')
            try:
                gt_output = json.loads(item['output'])
                gt_label = gt_output.get('label', 'unknown')
            except:
                gt_label = 'unknown'
            
            # Retrieve context if in RAG mode
            context = ""
            if retriever:
                context = retriever.get_context_for_classification(content, top_k=3)
            
            # Classify
            t0 = time.time()
            result = agent.classify(
                content=content,
                title=title,
                context=context
            )
            elapsed = time.time() - t0
            total_time += elapsed
            
            pred_label = result.get('label', 'unknown')
            is_correct = (pred_label.lower() == gt_label.lower())
            if is_correct:
                correct_count += 1
                
            predictions.append({
                "sample_id": i,
                "input_preview": content[:100],
                "ground_truth": gt_label,
                "prediction": pred_label,
                "correct": is_correct,
                "confidence": result.get('confidence'),
                "reasoning": result.get('reasoning'),
                "retrieved_context": context[:200] if context else "None",
                "inference_time": elapsed
            })
            
        accuracy = correct_count / len(samples) if samples else 0
        return {
            "mode": mode,
            "mode_name": RAG_MODES[mode]["name"],
            "accuracy": accuracy,
            "avg_time": total_time / len(samples) if samples else 0,
            "predictions": predictions
        }

# ─── Metrics & Analysis ───────────────────────────────────────────────────────

def analyze_results(results: Dict, output_dir: Path):
    """Generate comparative metrics for all modes."""
    summary = []
    for mode, data in results.items():
        summary.append({
            "Mode": RAG_MODES[mode]["name"],
            "Accuracy": data["accuracy"],
            "Avg Time (s)": data["avg_time"]
        })
    
    df = None
    if HAS_PANDAS:
        df = pd.DataFrame(summary)
        df.to_csv(output_dir / "comparison_summary.csv", index=False)
        print("\n📈 Comparison Summary:")
        print(df.to_string(index=False))
        
    if HAS_PLOT and df is not None:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Mode", y="Accuracy", data=df, palette="viridis")
        plt.title("RAG Ablation Study: Classification Accuracy", fontsize=14, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        for i, row in df.iterrows():
            plt.text(i, row['Accuracy'] + 0.02, f"{row['Accuracy']*100:.1f}%", ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_comparison.png", dpi=300)
        print(f"📊 Chart saved to {output_dir}/accuracy_comparison.png")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='RAG Ablation Study for Native Ads Detection')
    parser.add_argument('--dataset', type=str, default='../data/llm_dataset_finetuning_optimized.json')
    parser.add_argument('--num-samples', type=int, default=30, help='Number of test samples (per mode)')
    parser.add_argument('--model', type=str, default='google/gemma-3-12b-it', help='OpenRouter or local model path')
    parser.add_argument('--provider', type=str, default='openrouter', choices=['openrouter', 'openai', 'local'])
    parser.add_argument('--api-key', type=str, default=None)
    parser.add_argument('--embedding-model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--output-dir', type=str, default='rag_ablation_results')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY')
    if not api_key and args.provider in ['openai', 'openrouter']:
        print("❌ API key required for cloud providers via --api-key or OPENROUTER_API_KEY env.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("RAG ABLATION STUDY")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples: {args.num_samples}")
    logger.info("")

    # Load test data
    try:
        with open(args.dataset, 'r', encoding='utf-8') as f:
            data = json.load(f)
        import random
        random.seed(42)
        random.shuffle(data)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Initialize evaluator
    evaluator = RAGAblationEvaluator(
        model_name=args.model,
        provider=args.provider,
        api_key=api_key,
        embedding_model=args.embedding_model
    )
    
    # Run evaluation
    results = evaluator.evaluate(data, args.num_samples, output_dir)
    
    # Analyze and plot
    analyze_results(results, output_dir)
    
    logger.info("\n✅ RAG Ablation Study Complete!")
    logger.info(f"Full results stored in: {args.output_dir}/")

if __name__ == "__main__":
    main()
