"""
Full Pipeline Chain
Combines all agents and tools into a complete workflow
"""

from typing import Dict, Any, Optional
from langchain.chains import SequentialChain, TransformChain
import logging

from tools import WebScraperTool, TextCleanerTool, FeatureExtractorTool, SummarizerTool
from agents.classification_agent import ClassificationAgent
from agents.explanation_agent import ExplanationAgent

logger = logging.getLogger(__name__)


class FullPipelineChain:
    """
    Complete pipeline for native ads detection.
    URL → Scraping → Preprocessing → Retrieval → Classification → Explanation → Result
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        provider: str = "openai",
        api_key: Optional[str] = None,
        vectorstore: Optional[Any] = None,
        generate_explanation: bool = True
    ):
        """
        Initialize full pipeline.
        
        Args:
            model_name: LLM model name
            provider: LLM provider
            api_key: API key
            vectorstore: Optional vector store for RAG
            generate_explanation: Whether to generate detailed explanations
        """
        # Initialize tools
        self.web_scraper = WebScraperTool()
        self.text_cleaner = TextCleanerTool()
        self.feature_extractor = FeatureExtractorTool()
        self.summarizer = SummarizerTool()
        
        # Initialize classification agent
        self.classifier = ClassificationAgent(
            model_name=model_name,
            provider=provider,
            api_key=api_key
        )
        
        # Initialize explanation agent (optional)
        self.generate_explanation = generate_explanation
        if generate_explanation:
            self.explainer = ExplanationAgent(
                model_name=model_name,
                provider=provider,
                api_key=api_key
            )
        else:
            self.explainer = None
        
        self.vectorstore = vectorstore
        
        logger.info("Full Pipeline Chain initialized (with explanation support)")
    
    def run(self, url: str) -> Dict[str, Any]:
        """
        Run complete pipeline on a URL.
        
        Args:
            url: URL to analyze
            
        Returns:
            Complete analysis result
        """
        try:
            logger.info(f"Starting pipeline for: {url}")
            
            # Step 1: Web Scraping
            logger.info("[1/4] Scraping web content...")
            scraped_data = self.web_scraper.run(url)
            
            if 'error' in scraped_data:
                return {
                    'url': url,
                    'error': scraped_data['error'],
                    'status': 'failed'
                }
            
            # Step 2: Text Preprocessing
            logger.info("[2/4] Preprocessing text...")
            raw_text = scraped_data.get('text', '')
            cleaned_text = self.text_cleaner.run(raw_text)
            
            # Extract features
            features = self.feature_extractor.run(cleaned_text)
            
            # Create summary
            paragraphs = scraped_data.get('paragraphs', [])
            summary = self.summarizer.run(paragraphs)
            
            # Step 3: Retrieval (if vectorstore available)
            logger.info("[3/4] Retrieving context...")
            context = ""
            if self.vectorstore:
                try:
                    from tools.retrieval_tools import VectorSearchTool
                    retriever = VectorSearchTool(vectorstore=self.vectorstore)
                    retrieved_docs = retriever.run(cleaned_text[:500])
                    
                    if retrieved_docs:
                        context = "\n\n".join([
                            f"[Context {i+1}] {doc['content'][:200]}"
                            for i, doc in enumerate(retrieved_docs[:3])
                        ])
                except Exception as e:
                    logger.warning(f"Retrieval failed: {e}")
            
            # Step 4: Classification
            logger.info("[4/5] Classifying content...")
            classification = self.classifier.classify(
                content=cleaned_text,
                title=scraped_data.get('title', ''),
                summary=summary,
                context=context
            )
            
            # Step 5: Generate Explanation (if enabled)
            explanation = ""
            if self.generate_explanation and self.explainer:
                logger.info("[5/5] Generating explanation...")
                explanation = self.explainer.explain(
                    content=cleaned_text,
                    classification_result=classification
                )
            
            # Combine results
            result = {
                'url': url,
                'title': scraped_data.get('title', ''),
                'summary': summary,
                'classification': classification,
                'explanation': explanation,  # Added explanation
                'features': features,
                'metadata': {
                    **scraped_data.get('metadata', {}),
                    'pipeline_version': '1.0-langchain',
                    'has_context': bool(context),
                    'has_explanation': bool(explanation)
                },
                'status': 'success'
            }
            
            logger.info(f"Pipeline complete: {classification.get('label')}")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {
                'url': url,
                'error': str(e),
                'status': 'failed'
            }
    
    def run_batch(self, urls: list) -> list:
        """
        Run pipeline on multiple URLs.
        
        Args:
            urls: List of URLs to analyze
            
        Returns:
            List of results
        """
        results = []
        for i, url in enumerate(urls, 1):
            logger.info(f"Processing URL {i}/{len(urls)}")
            result = self.run(url)
            results.append(result)
        
        return results
