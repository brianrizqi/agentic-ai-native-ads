"""
Crawl Tools for News URL Discovery
"""

from typing import Optional, Type, List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForToolRun
import requests
from bs4 import BeautifulSoup
import logging
import re

logger = logging.getLogger(__name__)

class NewsCrawlerInput(BaseModel):
    """Input schema for NewsCrawlerTool."""
    source_url: str = Field(default="https://www.cnnindonesia.com/terkini", description="The news portal index URL to crawl")
    limit: int = Field(default=50, description="Number of news URLs to collect")

class NewsCrawlerTool(BaseTool):
    """Tool for discovering news article URLs from Indonesia portals."""
    
    name: str = "news_crawler"
    description: str = """
    Crawls a news portal's "latest news" or index page to extract article URLs.
    Supports Indonesian portals like CNN Indonesia.
    """
    args_schema: Type[BaseModel] = NewsCrawlerInput
    
    def _run(
        self,
        source_url: str = "https://www.cnnindonesia.com/terkini",
        limit: int = 50,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[str]:
        """Collect article URLs."""
        try:
            logger.info(f"Crawling news index: {source_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            urls = []
            
            # Simple crawling logic for CNN Indonesia (example)
            # CNN Indonesia pagination: /terkini/1, /terkini/2, etc.
            page = 1
            while len(urls) < limit:
                current_url = source_url if page == 1 else f"{source_url}/{page}"
                logger.debug(f"Fetching page {page}: {current_url}")
                
                response = requests.get(current_url, headers=headers, timeout=10)
                if response.status_code != 200:
                    break
                    
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # CNN Indonesia specific article links
                # Usually in <article> tags or links within a specific div
                links = soup.select('article a[href*="/nasional/"], article a[href*="/ekonomi/"], article a[href*="/teknologi/"]')
                
                for link in links:
                    href = link['href']
                    if href not in urls and href.startswith('http'):
                        urls.append(href)
                        if len(urls) >= limit:
                            break
                
                if not links: # No more links found
                    break
                    
                page += 1
                if page > 5: # Safety limit
                    break
            
            logger.info(f"Found {len(urls)} article URLs")
            return urls[:limit]
            
        except Exception as e:
            logger.error(f"News crawling error: {e}")
            return []
