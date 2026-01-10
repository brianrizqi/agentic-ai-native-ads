"""
Web Scraping Tools for LangChain
Provides tools for scraping web content using BeautifulSoup and Selenium
"""

from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForToolRun
import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


class WebScraperInput(BaseModel):
    """Input schema for WebScraperTool."""
    url: str = Field(description="The URL to scrape content from")


class WebScraperTool(BaseTool):
    """Tool for scraping web content using BeautifulSoup."""
    
    name: str = "web_scraper"
    description: str = """
    Scrapes content from a given URL and extracts text, title, and paragraphs.
    Input should be a valid URL string.
    Returns a dictionary with scraped content.
    """
    args_schema: Type[BaseModel] = WebScraperInput
    
    def _run(
        self,
        url: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        """Scrape content from URL."""
        try:
            logger.info(f"Scraping URL: {url}")
            
            # Fetch page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Extract main content
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract paragraphs
            paragraphs = []
            for p in soup.find_all('p'):
                p_text = p.get_text().strip()
                if len(p_text) > 20:  # Filter out short paragraphs
                    paragraphs.append(p_text)
            
            result = {
                'url': url,
                'title': title_text,
                'text': text,
                'paragraphs': paragraphs[:10],  # Limit to 10 paragraphs
                'metadata': {
                    'status_code': response.status_code,
                    'content_type': response.headers.get('content-type', ''),
                    'scraping_method': 'beautifulsoup'
                }
            }
            
            logger.info(f"Successfully scraped {len(text)} characters from {url}")
            return result
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return {
                'url': url,
                'error': str(e),
                'title': '',
                'text': '',
                'paragraphs': []
            }


class SeleniumScraperInput(BaseModel):
    """Input schema for SeleniumScraperTool."""
    url: str = Field(description="The URL to scrape with JavaScript rendering")
    wait_time: int = Field(default=3, description="Seconds to wait for page load")


class SeleniumScraperTool(BaseTool):
    """Tool for scraping JavaScript-heavy websites using Selenium."""
    
    name: str = "selenium_scraper"
    description: str = """
    Scrapes content from JavaScript-heavy websites using Selenium.
    Use this when regular scraping fails or for dynamic content.
    Input should be a URL and optional wait time.
    """
    args_schema: Type[BaseModel] = SeleniumScraperInput
    
    def _run(
        self,
        url: str,
        wait_time: int = 3,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        """Scrape content using Selenium."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            import time
            
            logger.info(f"Scraping with Selenium: {url}")
            
            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            
            # Create driver
            driver = webdriver.Chrome(options=chrome_options)
            
            try:
                driver.get(url)
                time.sleep(wait_time)  # Wait for JavaScript to load
                
                # Get page source
                page_source = driver.page_source
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(page_source, 'html.parser')
                
                # Extract content (same as WebScraperTool)
                title = soup.find('title')
                title_text = title.get_text().strip() if title else ""
                
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                paragraphs = []
                for p in soup.find_all('p'):
                    p_text = p.get_text().strip()
                    if len(p_text) > 20:
                        paragraphs.append(p_text)
                
                result = {
                    'url': url,
                    'title': title_text,
                    'text': text,
                    'paragraphs': paragraphs[:10],
                    'metadata': {
                        'scraping_method': 'selenium',
                        'wait_time': wait_time
                    }
                }
                
                logger.info(f"Successfully scraped {len(text)} characters with Selenium")
                return result
                
            finally:
                driver.quit()
                
        except ImportError:
            logger.error("Selenium not installed. Install with: pip install selenium")
            return {'error': 'Selenium not installed', 'url': url}
        except Exception as e:
            logger.error(f"Selenium scraping error: {e}")
            return {'error': str(e), 'url': url}
