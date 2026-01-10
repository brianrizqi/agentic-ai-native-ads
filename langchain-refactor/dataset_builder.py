"""
Dataset Builder for Native Ads Detection
Automates URL collection, processing, and dataset creation
"""

import logging
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

from tools import NewsCrawlerTool
from chains.full_pipeline_chain import FullPipelineChain

logger = logging.getLogger(__name__)

class DatasetBuilder:
    """
    Builder class to create labeled datasets for native ads detection.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        provider: str = "openrouter",
        api_key: Optional[str] = None
    ):
        """Initialize builder with existing pipeline."""
        self.pipeline = FullPipelineChain(
            model_name=model_name,
            provider=provider,
            api_key=api_key
        )
        self.crawler = NewsCrawlerTool()
        
    def collect_and_build(
        self,
        target_count: int = 50,
        output_name: str = "news_dataset",
        formats: List[str] = ["csv", "json"]
    ) -> Dict[str, str]:
        """
        Main function to collect news articles and build a dataset.
        
        Args:
            target_count: Number of articles to collect
            output_name: Base name for output files
            formats: List of output formats ('csv', 'json')
            
        Returns:
            Dictionary with paths to generated files
        """
        logger.info(f"Starting dataset collection for {target_count} articles...")
        
        # 1. Collect URLs
        urls = self.crawler.run({"limit": target_count})
        if not urls:
            logger.error("No article URLs found. Collection aborted.")
            return {}
            
        logger.info(f"Collected {len(urls)} URLs. Starting processing...")
        
        # 2. Process through pipeline
        results = []
        for i, url in enumerate(urls, 1):
            logger.info(f"[{i}/{len(urls)}] Processing: {url}")
            try:
                result = self.pipeline.run(url)
                if result.get('status') == 'success':
                    results.append(result)
                else:
                    logger.warning(f"Failed to process {url}: {result.get('error')}")
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                
        # 3. Save results
        output_paths = {}
        
        # Format for CSV
        if "csv" in formats:
            csv_path = f"data/{output_name}.csv"
            Path("data").mkdir(parents=True, exist_ok=True)
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['url', 'title', 'content_summary', 'label', 'confidence', 'reasoning'])
                for r in results:
                    writer.writerow([
                        r['url'],
                        r['title'],
                        r['summary'][:200].replace('\n', ' '),
                        r['classification']['label'],
                        f"{r['classification']['confidence']:.2f}",
                        r['classification']['reasoning']
                    ])
            output_paths['csv'] = csv_path
            logger.info(f"CSV dataset saved to {csv_path}")
            
        # Format for LLM Ready JSON (Instruction Tuning format)
        if "json" in formats:
            json_path = f"data/{output_name}_llm_ready.json"
            llm_data = []
            for r in results:
                llm_data.append({
                    "instruction": "Klasifikasikan apakah berita berikut adalah 'berita murni' atau 'native ads' (iklan terselubung).",
                    "input": f"Judul: {r['title']}\n\nKonten: {r['summary']}",
                    "output": f"Label: {r['classification']['label']}\nAlasan: {r['classification']['reasoning']}"
                })
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(llm_data, f, indent=2, ensure_ascii=False)
            output_paths['json'] = json_path
            logger.info(f"LLM-ready dataset saved to {json_path}")
            
        return output_paths

def build_dataset_from_instruction(
    instruction_text: str,
    api_key: Optional[str] = None
) -> Dict[str, str]:
    """
    High-level function to build a dataset based on user instruction.
    Input: "tolong kumpulkan 50 berita lalu jadikan dalam bentuk dataset csv atau llm ready dataset..."
    """
    # Simple regex to find numbers in instruction
    import re
    nums = re.findall(r'\d+', instruction_text)
    count = int(nums[0]) if nums else 50
    
    # Initialize and run
    builder = DatasetBuilder(api_key=api_key)
    return builder.collect_and_build(target_count=count)

if __name__ == "__main__":
    # Test run
    # import sys
    # api_key = "..."
    # build_dataset_from_instruction("tolong kumpulkan 5 berita", api_key=api_key)
    pass
