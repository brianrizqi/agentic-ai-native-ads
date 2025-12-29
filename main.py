"""
Main Orchestrator for Agentic AI System
Pipeline: Web Agent (Scraping) -> Preprocessing Agent -> Agentic AI
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add agents directory to path
sys.path.append(str(Path(__file__).parent / 'agents'))

from agents.web_agent import WebAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.retriever_agent import RetrieverAgent
from agents.llm_classifier_agent import LLMClassifierAgent
from agents.explanation_agent import ExplanationAgent
from agents.feedback_retrainer_agent import FeedbackReTrainerAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AgenticAISystem:
    """
    Main orchestrator for the Agentic AI System.
    Pipeline: Web Scraping -> Preprocessing -> Retrieval -> Classification -> Explanation -> Feedback
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Agentic AI System.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
        
        # Initialize agents
        logger.info("Initializing Agentic AI System...")
        
        # Web Agent
        self.web_agent = WebAgent(
            timeout=config.get('web_agent', {}).get('timeout', 30),
            max_retries=config.get('web_agent', {}).get('max_retries', 3)
        )
        
        # Preprocessing Agent
        self.preprocessing_agent = PreprocessingAgent(
            max_length=config.get('preprocessing', {}).get('max_length', 2000)
        )
        
        # Retriever Agent
        self.retriever_agent = RetrieverAgent(
            vector_db_path=config.get('retriever', {}).get('vector_db_path'),
            embedding_model=config.get('retriever', {}).get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            top_k=config.get('retriever', {}).get('top_k', 5)
        )
        
        # LLM Classifier Agent
        self.llm_classifier = LLMClassifierAgent(
            api_key=config.get('llm', {}).get('api_key', ''),
            model_name=config.get('llm', {}).get('model_name', 'gpt-4'),
            temperature=config.get('llm', {}).get('temperature', 0.3),
            max_tokens=config.get('llm', {}).get('max_tokens', 1000)
        )
        
        # Explanation Agent
        self.explanation_agent = ExplanationAgent(
            api_key=config.get('llm', {}).get('api_key', ''),
            model_name=config.get('llm', {}).get('model_name', 'gpt-4'),
            temperature=config.get('llm', {}).get('explanation_temperature', 0.7),
            max_tokens=config.get('llm', {}).get('explanation_max_tokens', 1500)
        )
        
        # Feedback/ReTrainer Agent
        self.feedback_agent = FeedbackReTrainerAgent(
            feedback_dir=config.get('feedback', {}).get('feedback_dir', 'data/feedback')
        )
        
        logger.info("Agentic AI System initialized successfully")
    
    def process_url(self, url: str, user_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a single URL through the entire pipeline.
        
        Args:
            url: URL to process
            user_feedback: Optional user feedback
            
        Returns:
            Complete results from all agents
        """
        logger.info(f"Processing URL: {url}")
        
        try:
            # Step 1: Web Agent - Scraping
            logger.info("Step 1: Web scraping...")
            scraped_data = self.web_agent.scrape(url)
            
            # Step 2: Preprocessing Agent
            logger.info("Step 2: Preprocessing...")
            preprocessed_data = self.preprocessing_agent.process(scraped_data)
            
            # Step 3: Retriever Agent
            logger.info("Step 3: Retrieving context...")
            retrieved_context = self.retriever_agent.retrieve(preprocessed_data)
            
            # Step 4: LLM Classifier Agent
            logger.info("Step 4: Classification...")
            classification = self.llm_classifier.classify(preprocessed_data, retrieved_context)
            
            # Step 5: Explanation Agent
            logger.info("Step 5: Generating explanation...")
            explanation = self.explanation_agent.explain(
                classification, preprocessed_data, retrieved_context
            )
            
            # Step 6: Feedback Agent
            logger.info("Step 6: Collecting feedback...")
            feedback_analysis = self.feedback_agent.collect_feedback(
                classification, explanation, preprocessed_data, user_feedback
            )
            
            # Compile results
            results = {
                'success': True,
                'url': url,
                'scraped_data': {
                    'title': scraped_data.get('title'),
                    'word_count': scraped_data.get('word_count'),
                    'metadata': scraped_data.get('metadata')
                },
                'preprocessed_data': {
                    'cleaned_text_length': len(preprocessed_data.get('cleaned_text', '')),
                    'token_count': preprocessed_data.get('metadata', {}).get('token_count'),
                    'summary': preprocessed_data.get('summary')
                },
                'retrieved_context': {
                    'num_documents': len(retrieved_context),
                    'documents': retrieved_context
                },
                'classification': classification,
                'explanation': explanation,
                'feedback_analysis': feedback_analysis
            }
            
            logger.info("Processing complete!")
            return results
            
        except Exception as e:
            logger.error(f"Error processing URL: {str(e)}")
            return {
                'success': False,
                'url': url,
                'error': str(e)
            }
    
    def process_multiple_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple URLs.
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of results
        """
        logger.info(f"Processing {len(urls)} URLs")
        results = []
        
        for url in urls:
            result = self.process_url(url)
            results.append(result)
        
        return results
    
    def add_knowledge_base_documents(self, documents: List[str]):
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of document texts
        """
        logger.info(f"Adding {len(documents)} documents to knowledge base")
        self.retriever_agent.add_documents(documents)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            'retriever_stats': self.retriever_agent.get_stats(),
            'feedback_summary': self.feedback_agent.get_feedback_summary()
        }


def main():
    """Main execution function."""
    import json
    
    # Load configuration
    config_path = Path(__file__).parent / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize system
    system = AgenticAISystem(config)
    
    # Example: Add some documents to knowledge base
    sample_documents = [
        "Artificial Intelligence is transforming various industries including healthcare, finance, and education.",
        "Machine learning models require large amounts of data for training and validation.",
        "Natural language processing enables computers to understand and generate human language."
    ]
    system.add_knowledge_base_documents(sample_documents)
    
    # Example: Process a URL
    example_url = "https://example.com/article"
    
    print("\n" + "="*80)
    print(f"Processing: {example_url}")
    print("="*80 + "\n")
    
    results = system.process_url(example_url)
    
    if results['success']:
        print("\n--- Results ---")
        print(f"Title: {results['scraped_data']['title']}")
        print(f"Classification: {results['classification']['label']}")
        print(f"Confidence: {results['classification']['confidence']:.2%}")
        print(f"\nExplanation Summary:")
        print(results['explanation']['summary'])
        print(f"\nFeedback Quality: {results['feedback_analysis']['quality_level']}")
    else:
        print(f"\nError: {results['error']}")
    
    # Get system stats
    print("\n--- System Statistics ---")
    stats = system.get_system_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
