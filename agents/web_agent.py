"""
Web Agent - Web Scraping Module
Handles web scraping from URLs and extracts content.
"""

import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import time

logger = logging.getLogger(__name__)


class WebAgent:
    """
    Web Agent yang melakukan scraping dari URL dan mengekstrak konten.
    """
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """
        Initialize Web Agent.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        logger.info("Web Agent initialized")
    
    def scrape(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from a URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary containing scraped content and metadata
        """
        logger.info(f"Starting scraping: {url}")
        
        # Validate URL
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")
        
        # Scrape with retries
        content = None
        for attempt in range(self.max_retries):
            try:
                content = self._fetch_content(url)
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        # Extract structured data
        extracted_data = self._extract_data(content, url)
        
        logger.info(f"Scraping complete: {len(extracted_data['text'])} characters extracted")
        return extracted_data
    
    def scrape_multiple(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of scraped content dictionaries
        """
        logger.info(f"Scraping {len(urls)} URLs")
        results = []
        
        for url in urls:
            try:
                result = self.scrape(url)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {str(e)}")
                results.append({
                    'url': url,
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _fetch_content(self, url: str) -> str:
        """Fetch raw HTML content from URL."""
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text
    
    def _extract_data(self, html_content: str, url: str) -> Dict[str, Any]:
        """
        Extract structured data from HTML content.
        
        Args:
            html_content: Raw HTML content
            url: Source URL
            
        Returns:
            Extracted and structured data
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
            script.decompose()
        
        # Extract text
        text = soup.get_text(separator=' ', strip=True)
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text(strip=True) if title else ''
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc.get('content', '') if meta_desc else ''
        
        # Extract headings
        headings = []
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                headings.append({
                    'level': i,
                    'text': heading.get_text(strip=True)
                })
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            links.append({
                'text': link.get_text(strip=True),
                'href': link['href']
            })
        
        # Extract paragraphs
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
        
        return {
            'url': url,
            'title': title_text,
            'description': description,
            'text': text,
            'paragraphs': paragraphs,
            'headings': headings,
            'links': links[:20],  # Limit to first 20 links
            'word_count': len(text.split()),
            'char_count': len(text),
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'metadata': {
                'domain': urlparse(url).netloc,
                'scraping_method': 'beautifulsoup',
                'num_headings': len(headings),
                'num_paragraphs': len(paragraphs),
                'num_links': len(links)
            }
        }
