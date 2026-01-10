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
    limit: int = Field(default=50, description="Total number of news URLs to collect")
    sources: Optional[List[str]] = Field(default=None, description="List of portal base URLs to crawl")

class NewsCrawlerTool(BaseTool):
    """Tool for discovering news article URLs from multiple Indonesian portals."""
    
    name: str = "news_crawler"
    description: str = """
    Crawls multiple news portals to extract article URLs.
    Supports CNN Indonesia, Detik, Kompas, Viva, Tempo, and Sindonews.
    Returns a list of unique article URLs.
    """
    args_schema: Type[BaseModel] = NewsCrawlerInput
    
    def _run(
        self,
        limit: int = 50,
        sources: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[str]:
        """Collect article URLs from various sources."""
        try:
            # Default sources if none provided
            if not sources:
                sources = [
                    "https://www.cnnindonesia.com/tag/laporan-interaktif",
                    "https://news.detik.com/indeks",
                    "https://www.viva.co.id/terpopuler",
                    "https://indeks.kompas.com/terpopuler",
                    "https://www.tempo.co/indeks",
                    "https://www.sindonews.com/indeks"
                ]
            
            logger.info(f"Crawling {len(sources)} sources for {limit} articles...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            }
            
            all_urls = []
            urls_per_source = max(1, limit // len(sources)) + 5
            
            for source in sources:
                if len(all_urls) >= limit:
                    break
                    
                logger.info(f"Crawling source: {source}")
                try:
                    response = requests.get(source, headers=headers, timeout=10)
                    if response.status_code != 200:
                        logger.warning(f"Failed to fetch {source}: {response.status_code}")
                        continue
                        
                    soup = BeautifulSoup(response.content, 'html.parser')
                    source_urls = []
                    
                    # 1. Domain-specific selectors
                    if "cnnindonesia.com" in source:
                        # CNN uses specific link patterns
                        links = soup.find_all('a', href=re.compile(r'/(nasional|ekonomi|teknologi|internasional|hiburan|gaya-hidup|olahraga|otomotif)/20\d{6}'))
                    elif "detik.com" in source:
                        links = soup.select('article a[href*="detik.com/news/"]')
                    elif "kompas.com" in source:
                        links = soup.select('.article__list__title a')
                    elif "viva.co.id" in source:
                        links = soup.select('.article-list a')
                    elif "tempo.co" in source:
                        links = soup.select('.card-box a[href*="tempo.co/read/"]')
                    elif "sindonews.com" in source:
                        links = soup.select('.m-link a')
                    else:
                        # Fallback for generic portals
                        links = soup.find_all('a', href=True)
                    
                    for link in links:
                        href = link['href']
                        
                        # Clean relative URLs
                        if href.startswith('/'):
                            # Try to construct absolute URL
                            domain = re.match(r'(https?://[^/]+)', source).group(1)
                            href = domain + href
                            
                        # Basic filtering
                        if not href.startswith('http'):
                            continue
                        
                        # Avoid duplicates and non-article links
                        if (href not in all_urls and 
                            href not in source_urls and 
                            not any(x in href for x in ['/tag/', '/search/', '/video/', '/foto/', '/indeks', '/author/']) and
                            (len(href.split('/')) > 4 or 'read' in href)):
                            
                            source_urls.append(href)
                            if len(all_urls) + len(source_urls) >= limit or len(source_urls) >= urls_per_source:
                                break
                    
                    all_urls.extend(source_urls)
                    logger.info(f"Found {len(source_urls)} urls from {source}")
                    
                except Exception as e:
                    logger.error(f"Error crawling {source}: {e}")
                    continue
            
            # Final unique set and limit
            final_urls = list(dict.fromkeys(all_urls))[:limit]
            logger.info(f"Total unique URLs found: {len(final_urls)}")
            return final_urls
            
        except Exception as e:
            logger.error(f"News crawling master error: {e}")
            return []
