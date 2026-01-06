"""
Utility Functions for LangChain Refactored System
"""

import logging
from pathlib import Path
from typing import Any, Dict


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid, False otherwise
    """
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None


def format_classification_result(result: Dict[str, Any]) -> str:
    """
    Format classification result for display.
    
    Args:
        result: Classification result dictionary
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("="*80)
    lines.append("CLASSIFICATION RESULT")
    lines.append("="*80)
    
    if result.get('status') == 'failed':
        lines.append(f"Error: {result.get('error', 'Unknown error')}")
        return "\n".join(lines)
    
    lines.append(f"URL: {result.get('url', 'N/A')}")
    lines.append(f"Title: {result.get('title', 'N/A')}")
    lines.append("")
    
    classification = result.get('classification', {})
    lines.append(f"Label: {classification.get('label', 'unknown')}")
    lines.append(f"Confidence: {classification.get('confidence', 0):.2%}")
    lines.append(f"Reasoning: {classification.get('reasoning', 'N/A')}")
    lines.append("")
    
    features = result.get('features', {})
    if features:
        lines.append("Features:")
        lines.append(f"  Word Count: {features.get('word_count', 0)}")
        lines.append(f"  Sentence Count: {features.get('sentence_count', 0)}")
        lines.append(f"  Lexical Diversity: {features.get('lexical_diversity', 0):.2f}")
    
    lines.append("="*80)
    return "\n".join(lines)


def save_results_to_json(results: list, output_path: str) -> None:
    """
    Save results to JSON file.
    
    Args:
        results: List of result dictionaries
        output_path: Output file path
    """
    import json
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_urls_from_file(file_path: str) -> list:
    """
    Load URLs from text file (one per line).
    
    Args:
        file_path: Path to file containing URLs
        
    Returns:
        List of URLs
    """
    with open(file_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    return urls
