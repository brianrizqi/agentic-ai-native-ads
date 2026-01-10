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

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from prompts.classification_prompts import discovery_prompt

class NewsCrawlerInput(BaseModel):
    """Input schema for NewsCrawlerTool."""
    limit: int = Field(default=50, description="Total number of news URLs to collect")
    sources: Optional[List[str]] = Field(default=None, description="List of portal base URLs to crawl")

class NewsCrawlerTool(BaseTool):
    """AI-driven tool for discovering news article URLs from multiple Indonesian portals."""
    
    name: str = "news_crawler"
    description: str = """
    Intelligently identifies news article URLs from portal index pages using AI.
    Works by fetching all links and letting the LLM decide which ones are news.
    """
    args_schema: Type[BaseModel] = NewsCrawlerInput
    llm: Any = None
    
    def __init__(self, llm=None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        if self.llm:
            self.discovery_chain = discovery_prompt | self.llm | StrOutputParser()
    
    def _run(
        self,
        limit: int = 50,
        sources: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[str]:
        """Collect article URLs using AI-driven discovery."""
        try:
            if not sources:
                sources = [
                    "https://www.cnnindonesia.com/terkini",
                    "https://news.detik.com/indeks",
                    "https://www.viva.co.id/berita/terbaru",
                    "https://news.kompas.com/indeks"
                ]
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            all_discovered_urls = []
            
            for source in sources:
                if len(all_discovered_urls) >= limit:
                    break
                    
                logger.info(f"AI Discovery on: {source}")
                try:
                    response = requests.get(source, headers=headers, timeout=10)
                    if response.status_code != 200:
                        continue
                        
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # 1. Ambil semua link dengan teksnya
                    potential_links = []
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        text = link.get_text().strip()
                        
                        # Clean relative URLs
                        if href.startswith('/'):
                            domain = re.match(r'(https?://[^/]+)', source).group(1)
                            href = domain + href
                        
                        if href.startswith('http') and len(text) > 10:
                            potential_links.append(f"{text}: {href}")
                    
                    # 2. Limit potential links to avoid context overflow
                    sample_links = "\n".join(potential_links[:100])
                    
                    # 3. Use AI to identify news links
                    if self.llm:
                        logger.info(f"LLM identifying news from {len(potential_links)} candidates...")
                        ai_response = self.discovery_chain.invoke({
                            "source": source,
                            "links": sample_links,
                            "limit": max(5, limit // len(sources))
                        })
                        
                        # Parse JSON from AI response (very simple extract)
                        import json
                        match = re.search(r'\[.*\]', ai_response, re.DOTALL)
                        if match:
                            found_urls = json.loads(match.group(0))
                            all_discovered_urls.extend(found_urls)
                            logger.info(f"AI found {len(found_urls)} articles on {source}")
                        else:
                            logger.warning("AI response didn't contain a JSON list")
                    else:
                        # Fallback to simple regex if no LLM
                        logger.warning("No LLM provided to NewsCrawlerTool, using regex fallback")
                        found_urls = [l.split(': ')[1] for l in potential_links if '/read/' in l or '/20' in l]
                        all_discovered_urls.extend(found_urls[:10])
                        
                except Exception as e:
                    logger.error(f"Error in discovery for {source}: {e}")
                    continue
            
            return list(dict.fromkeys(all_discovered_urls))[:limit]
            
        except Exception as e:
            logger.error(f"Master discovery error: {e}")
            return []
