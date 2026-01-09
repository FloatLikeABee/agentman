"""
Advanced Web Crawler Service
Uses best-in-class scraping libraries to extract content from websites,
process with AI, and save to RAG collections.
"""
import os
import json
import logging
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Set, List
from urllib.parse import urljoin, urlparse
from collections import deque

import requests
from bs4 import BeautifulSoup

from .config import settings
from .rag_system import RAGSystem
from .llm_factory import LLMFactory, LLMProvider
from .models import RAGDataInput, DataFormat

# Try to import Playwright for advanced scraping (optional)
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Try to import Selenium as alternative (optional)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


class CrawlerService:
    """Advanced web crawler service with AI-powered content extraction"""
    
    def __init__(self, rag_system: RAGSystem):
        self.logger = logging.getLogger(__name__)
        self.rag_system = rag_system
        self.data_dir = Path(settings.data_directory) / "scraped_content"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def scrape_url(self, url: str, use_js: bool = False, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Scrape a URL using the best available method.
        
        Args:
            url: URL to scrape
            use_js: Whether to use JavaScript rendering (Playwright/Selenium)
            headers: Optional custom HTTP headers (e.g., for authentication)
        
        Returns:
            Dictionary with scraped content
        """
        self.logger.info(f"[Crawler] Scraping URL: {url} (JS: {use_js})")
        
        # Try Playwright first if JS rendering is needed
        if use_js and PLAYWRIGHT_AVAILABLE:
            try:
                return self._scrape_with_playwright(url, headers=headers)
            except Exception as e:
                self.logger.warning(f"Playwright scraping failed: {e}, falling back to requests")
        
        # Try Selenium as alternative for JS rendering
        if use_js and SELENIUM_AVAILABLE:
            try:
                return self._scrape_with_selenium(url, headers=headers)
            except Exception as e:
                self.logger.warning(f"Selenium scraping failed: {e}, falling back to requests")
        
        # Fallback to requests + BeautifulSoup
        return self._scrape_with_requests(url, headers=headers)
    
    def _scrape_with_playwright(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Scrape using Playwright (handles JavaScript-rendered content)"""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            
            # Set headers if provided
            if headers:
                context.set_extra_http_headers(headers)
            
            page = context.new_page()
            
            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
                # Wait for content to load
                page.wait_for_timeout(2000)
                
                # Get page content
                html = page.content()
                title = page.title()
                
                browser.close()
                
                return self._parse_html(html, url, title)
            except Exception as e:
                browser.close()
                raise
    
    def _scrape_with_selenium(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Scrape using Selenium (handles JavaScript-rendered content)"""
        options = ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        driver = webdriver.Chrome(options=options)
        try:
            # Set headers if provided (Selenium doesn't support custom headers directly,
            # but we can use execute_cdp_cmd for Chrome DevTools Protocol)
            if headers:
                # Note: Selenium header support is limited, this is a workaround
                driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                    "userAgent": headers.get('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
                })
            
            driver.get(url)
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2)  # Additional wait for JS content
            
            html = driver.page_source
            title = driver.title
            
            return self._parse_html(html, url, title)
        finally:
            driver.quit()
    
    def _scrape_with_requests(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Scrape using requests + BeautifulSoup (fast, but no JS rendering)"""
        default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Merge custom headers with defaults (custom headers take precedence)
        if headers:
            default_headers.update(headers)
        
        response = requests.get(url, headers=default_headers, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        return self._parse_html(response.content, url)
    
    def _parse_html(self, html_content: Any, url: str, title: Optional[str] = None) -> Dict[str, Any]:
        """Parse HTML content and extract structured data"""
        if isinstance(html_content, bytes):
            soup = BeautifulSoup(html_content, 'html.parser')
        else:
            soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
            element.decompose()
        
        # Extract title
        if not title:
            title = soup.title.string.strip() if soup.title else ""
        
        # Extract headings with hierarchy
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f'h{level}'):
                text = heading.get_text().strip()
                if text:
                    headings.append({"level": level, "text": text})
        
        # Extract paragraphs
        paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
        
        # Extract lists
        lists = []
        for ul in soup.find_all(['ul', 'ol']):
            items = [li.get_text().strip() for li in ul.find_all('li') if li.get_text().strip()]
            if items:
                lists.append(items)
        
        # Extract links
        links = []
        for a in soup.find_all('a', href=True):
            link_text = a.get_text().strip()
            href = a.get('href', '')
            if link_text and href:
                full_url = urljoin(url, href)
                links.append({"text": link_text, "url": full_url})
        
        # Extract main content (prioritize semantic HTML5 elements)
        main_content = ""
        for selector in ['main', 'article', '[role="main"]', '.content', '#content', '.main-content', 'body']:
            content_area = soup.select_one(selector)
            if content_area:
                main_content = content_area.get_text(separator='\n', strip=True)
                break
        
        if not main_content:
            main_content = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in main_content.splitlines() if line.strip()]
        main_content = '\n'.join(lines)
        
        return {
            "url": url,
            "title": title,
            "headings": headings[:50],
            "paragraphs": paragraphs[:100],
            "lists": lists[:30],
            "links": links[:200],
            "main_content": main_content[:200000],  # Limit to 200KB
            "scraped_at": datetime.now().isoformat()
        }
    
    def save_raw_content(self, raw_data: Dict[str, Any], url: str) -> Path:
        """Save raw scraped content to file"""
        safe_filename = urlparse(url).netloc.replace('.', '_') + '_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_file_path = self.data_dir / f"{safe_filename}_raw.json"
        
        with open(raw_file_path, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"[Crawler] Raw content saved to: {raw_file_path}")
        return raw_file_path
    
    def extract_with_ai(
        self, 
        raw_data: Dict[str, Any], 
        url: str,
        llm_provider: str = None,
        model: str = None
    ) -> Dict[str, Any]:
        """
        Use AI to extract and abstract useful information from scraped content.
        Also generates collection name and description.
        """
        self.logger.info(f"[Crawler] Extracting content with AI...")
        
        # Determine LLM provider
        provider_str = (llm_provider or settings.default_llm_provider).lower().strip()
        
        if provider_str == "gemini":
            provider = LLMProvider.GEMINI
            api_key = settings.gemini_api_key
            model = model or settings.gemini_default_model
        elif provider_str == "qwen":
            provider = LLMProvider.QWEN
            api_key = settings.qwen_api_key
            model = model or settings.qwen_default_model
        else:
            # Fallback to default
            provider = LLMProvider.GEMINI
            api_key = settings.gemini_api_key
            model = model or settings.gemini_default_model
        
        llm_caller = LLMFactory.create_caller(
            provider=provider,
            api_key=api_key,
            model=model
        )
        
        # Prepare content for AI (limit size)
        content_for_ai = {
            "url": url,
            "title": raw_data.get("title", ""),
            "headings": raw_data.get("headings", [])[:30],
            "key_paragraphs": raw_data.get("paragraphs", [])[:50],
            "main_content_preview": raw_data.get("main_content", "")[:40000]  # 40KB for AI
        }
        
        raw_data_str = json.dumps(content_for_ai, indent=2, ensure_ascii=False)
        
        # Enhanced prompt for extraction AND collection naming
        extract_prompt = f"""You are an expert data extraction and knowledge base specialist. Analyze the scraped website content and:

1. Extract useful, valuable information in structured format
2. Generate an appropriate collection name (short, descriptive, 2-4 words)
3. Generate a collection description (1-2 sentences explaining what this collection contains)

Website URL: {url}

SCRAPED CONTENT:
{raw_data_str}

INSTRUCTIONS:
- Extract the most important, factual information
- Remove navigation, ads, cookie notices, and noise
- Create concise summaries while preserving important details
- Organize information into clear, searchable categories
- Generate a meaningful collection name based on the content
- Generate a helpful description for the collection

OUTPUT FORMAT - Return ONLY valid JSON (no markdown, no explanations):
{{
    "collection_name": "short_descriptive_name",
    "collection_description": "Brief description of what this collection contains",
    "title": "Clear, descriptive title of the page",
    "summary": "A comprehensive 2-3 sentence summary of the main content",
    "main_topics": ["topic1", "topic2", "topic3", ...],
    "key_points": [
        "Important point 1 with context",
        "Important point 2 with context",
        ...
    ],
    "detailed_content": "Well-organized, detailed content. Remove all navigation, ads, and noise. Keep only valuable information.",
    "entities": {{
        "people": ["person1", "person2", ...],
        "organizations": ["org1", "org2", ...],
        "locations": ["location1", "location2", ...],
        "dates": ["date1", "date2", ...]
    }},
    "metadata": {{
        "source_url": "{url}",
        "extracted_at": "{datetime.now().isoformat()}",
        "content_type": "article|product|documentation|news|other"
    }}
}}

IMPORTANT: 
- Return ONLY the JSON object, no markdown code blocks, no explanations
- collection_name should be short, lowercase, use underscores (e.g., "yahoo_news", "tech_articles")
- Ensure all text is properly escaped for JSON
- Focus on factual, useful information"""
        
        # Call AI with timeout
        result_container = {'data': None, 'error': None, 'completed': False}
        
        def call_llm():
            try:
                start_time = time.time()
                result_container['data'] = llm_caller.generate(extract_prompt)
                elapsed = time.time() - start_time
                self.logger.info(f"[Crawler] AI extraction completed in {elapsed:.2f} seconds")
                result_container['completed'] = True
            except Exception as e:
                result_container['error'] = e
                result_container['completed'] = True
        
        thread = threading.Thread(target=call_llm)
        thread.daemon = True
        thread.start()
        
        max_timeout = min(settings.api_timeout, 120)  # Up to 2 minutes
        thread.join(timeout=max_timeout)
        
        if not result_container['completed']:
            raise TimeoutError(f"AI processing timed out after {max_timeout} seconds")
        
        if result_container['error']:
            raise result_container['error']
        
        extracted_data_str = result_container['data']
        if not extracted_data_str:
            raise ValueError("AI returned empty response")
        
        # Parse JSON response
        cleaned_response = extracted_data_str.strip()
        if '```json' in cleaned_response:
            cleaned_response = cleaned_response.split('```json')[1].split('```')[0].strip()
        elif '```' in cleaned_response:
            parts = cleaned_response.split('```')
            for part in parts:
                part = part.strip()
                if part.startswith('{') and part.endswith('}'):
                    cleaned_response = part
                    break
        
        extracted_data = json.loads(cleaned_response)
        
        # Ensure required fields
        if 'collection_name' not in extracted_data:
            # Generate fallback name
            domain = urlparse(url).netloc.replace('.', '_')
            extracted_data['collection_name'] = f"{domain}_content"
        
        if 'collection_description' not in extracted_data:
            extracted_data['collection_description'] = f"Content extracted from {url}"
        
        if 'title' not in extracted_data:
            extracted_data['title'] = raw_data.get('title', 'Untitled')
        
        if 'summary' not in extracted_data:
            extracted_data['summary'] = 'Content extracted from website'
        
        if 'metadata' not in extracted_data:
            extracted_data['metadata'] = {}
        
        extracted_data['metadata']['source_url'] = url
        extracted_data['metadata']['extracted_at'] = datetime.now().isoformat()
        
        return extracted_data
    
    def save_extracted_content(self, extracted_data: Dict[str, Any], url: str) -> Path:
        """Save AI-extracted content to file"""
        safe_filename = urlparse(url).netloc.replace('.', '_') + '_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        extracted_file_path = self.data_dir / f"{safe_filename}_extracted.json"
        
        with open(extracted_file_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"[Crawler] Extracted data saved to: {extracted_file_path}")
        return extracted_file_path
    
    def save_to_rag(
        self, 
        extracted_data: Dict[str, Any], 
        url: str,
        collection_name: Optional[str] = None,
        collection_description: Optional[str] = None
    ) -> bool:
        """Save extracted data to RAG collection"""
        if not self.rag_system:
            raise ValueError("RAG system not available")
        
        # Use AI-generated name/description if not provided
        final_collection_name = collection_name or extracted_data.get('collection_name', 'web_content')
        final_description = collection_description or extracted_data.get('collection_description', f"Content from {url}")
        
        self.logger.info(f"[Crawler] Saving to RAG collection: {final_collection_name}")
        
        rag_input = RAGDataInput(
            name=f"crawled_{final_collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=final_description,
            format=DataFormat.JSON,
            content=json.dumps(extracted_data, indent=2, ensure_ascii=False),
            tags=["crawled", "web", urlparse(url).netloc, "ai-extracted"],
            metadata={
                "source_url": url,
                "crawled_at": datetime.now().isoformat(),
                "title": extracted_data.get("title", ""),
                "content_type": extracted_data.get("metadata", {}).get("content_type", "other")
            }
        )
        
        return self.rag_system.add_data_to_collection(final_collection_name, rag_input)
    
    def _crawl_recursively(
        self,
        start_url: str,
        use_js: bool = False,
        headers: Optional[Dict[str, str]] = None,
        max_depth: int = 3,
        max_pages: int = 50,
        same_domain_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Recursively crawl a website following links.
        
        Returns:
            List of scraped page data dictionaries
        """
        visited: Set[str] = set()
        pages_data: List[Dict[str, Any]] = []
        base_domain = urlparse(start_url).netloc
        
        # Queue: (url, depth)
        queue = deque([(start_url, 0)])
        visited.add(start_url)
        
        while queue and len(pages_data) < max_pages:
            current_url, depth = queue.popleft()
            
            if depth > max_depth:
                continue
            
            try:
                self.logger.info(f"[Crawler] Crawling page {len(pages_data) + 1}/{max_pages}: {current_url} (depth: {depth})")
                page_data = self.scrape_url(current_url, use_js=use_js, headers=headers)
                pages_data.append(page_data)
                
                # Extract links for next level if not at max depth
                if depth < max_depth and len(pages_data) < max_pages:
                    links = page_data.get("links", [])
                    for link_info in links:
                        link_url = link_info.get("url", "")
                        if not link_url:
                            continue
                        
                        # Normalize URL
                        parsed_link = urlparse(link_url)
                        # Remove fragment and query params for comparison
                        normalized_link = f"{parsed_link.scheme}://{parsed_link.netloc}{parsed_link.path}"
                        
                        # Check if we should follow this link
                        if normalized_link in visited:
                            continue
                        
                        # Check domain restriction
                        if same_domain_only and parsed_link.netloc != base_domain:
                            continue
                        
                        # Only follow http/https links
                        if parsed_link.scheme not in ['http', 'https']:
                            continue
                        
                        # Avoid common non-content URLs
                        skip_patterns = [
                            '.pdf', '.zip', '.jpg', '.jpeg', '.png', '.gif', '.svg',
                            '.css', '.js', '.xml', '.rss', '.atom',
                            'mailto:', 'tel:', 'javascript:', '#'
                        ]
                        if any(pattern in link_url.lower() for pattern in skip_patterns):
                            continue
                        
                        visited.add(normalized_link)
                        queue.append((link_url, depth + 1))
                        
                        if len(pages_data) >= max_pages:
                            break
                            
            except Exception as e:
                self.logger.warning(f"[Crawler] Failed to crawl {current_url}: {e}")
                continue
        
        self.logger.info(f"[Crawler] Recursive crawl completed: {len(pages_data)} pages crawled, {len(visited)} URLs visited")
        return pages_data
    
    def _merge_pages_data(self, pages_data: List[Dict[str, Any]], start_url: str) -> Dict[str, Any]:
        """Merge data from multiple pages into a single structure"""
        if not pages_data:
            return {}
        
        if len(pages_data) == 1:
            return pages_data[0]
        
        # Merge all pages
        merged = {
            "url": start_url,
            "title": pages_data[0].get("title", "Multi-page Content"),
            "headings": [],
            "paragraphs": [],
            "lists": [],
            "links": [],
            "main_content": "",
            "scraped_at": datetime.now().isoformat(),
            "pages_count": len(pages_data),
            "pages": []
        }
        
        all_links = set()
        for page_data in pages_data:
            # Collect headings
            merged["headings"].extend(page_data.get("headings", []))
            
            # Collect paragraphs
            merged["paragraphs"].extend(page_data.get("paragraphs", []))
            
            # Collect lists
            merged["lists"].extend(page_data.get("lists", []))
            
            # Collect unique links
            for link in page_data.get("links", []):
                link_url = link.get("url", "")
                if link_url:
                    all_links.add(link_url)
            
            # Merge main content
            page_content = page_data.get("main_content", "")
            if page_content:
                merged["main_content"] += f"\n\n--- Page: {page_data.get('url', 'Unknown')} ---\n\n{page_content}"
            
            # Store page metadata
            merged["pages"].append({
                "url": page_data.get("url", ""),
                "title": page_data.get("title", ""),
                "scraped_at": page_data.get("scraped_at", "")
            })
        
        # Convert links set back to list format
        merged["links"] = [{"url": url, "text": ""} for url in sorted(all_links)]
        
        # Limit sizes
        merged["headings"] = merged["headings"][:200]
        merged["paragraphs"] = merged["paragraphs"][:500]
        merged["lists"] = merged["lists"][:100]
        merged["links"] = merged["links"][:500]
        merged["main_content"] = merged["main_content"][:500000]  # 500KB limit
        
        return merged
    
    def crawl_and_save(
        self, 
        url: str, 
        use_js: bool = False,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        collection_name: Optional[str] = None,
        collection_description: Optional[str] = None,
        follow_links: bool = False,
        max_depth: int = 3,
        max_pages: int = 50,
        same_domain_only: bool = True,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Complete crawler workflow: scrape, extract with AI, save to files and RAG.
        
        Returns:
            Dictionary with results including collection_name, file paths, etc.
        """
        try:
            # Step 1: Scrape (recursively if enabled)
            if follow_links:
                self.logger.info(f"[Crawler] Step 1/4: Recursively crawling {url} (max_depth={max_depth}, max_pages={max_pages})...")
                pages_data = self._crawl_recursively(
                    url, 
                    use_js=use_js, 
                    headers=headers,
                    max_depth=max_depth, 
                    max_pages=max_pages,
                    same_domain_only=same_domain_only
                )
                raw_data = self._merge_pages_data(pages_data, url)
                total_links_found = len(raw_data.get("links", []))
            else:
                self.logger.info(f"[Crawler] Step 1/4: Scraping {url}...")
                raw_data = self.scrape_url(url, use_js=use_js, headers=headers)
                total_links_found = len(raw_data.get("links", []))
            
            # Step 2: Save raw content
            self.logger.info(f"[Crawler] Step 2/4: Saving raw content...")
            raw_file_path = self.save_raw_content(raw_data, url)
            
            # Step 3: Extract with AI (includes generating collection name/description)
            self.logger.info(f"[Crawler] Step 3/4: Extracting with AI...")
            extracted_data = self.extract_with_ai(raw_data, url, llm_provider, model)
            
            # Step 4: Save extracted content
            self.logger.info(f"[Crawler] Step 4/4: Saving extracted content...")
            extracted_file_path = self.save_extracted_content(extracted_data, url)
            
            # Step 5: Save to RAG (use AI-generated name if not provided)
            final_collection_name = collection_name or extracted_data.get('collection_name')
            final_description = collection_description or extracted_data.get('collection_description')
            
            success = self.save_to_rag(
                extracted_data, 
                url,
                collection_name=final_collection_name,
                collection_description=final_description
            )
            
            pages_crawled = raw_data.get("pages_count", 1) if follow_links else 1
            
            if success:
                return {
                    "success": True,
                    "url": url,
                    "collection_name": final_collection_name,
                    "collection_description": final_description,
                    "raw_file": str(raw_file_path),
                    "extracted_file": str(extracted_file_path),
                    "extracted_data": {
                        "title": extracted_data.get("title", ""),
                        "summary": extracted_data.get("summary", ""),
                        "main_topics_count": len(extracted_data.get("main_topics", [])),
                        "key_points_count": len(extracted_data.get("key_points", []))
                    },
                    "pages_crawled": pages_crawled,
                    "total_links_found": total_links_found
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to save to RAG collection",
                    "url": url,
                    "raw_file": str(raw_file_path),
                    "extracted_file": str(extracted_file_path),
                    "pages_crawled": pages_crawled,
                    "total_links_found": total_links_found
                }
                
        except Exception as e:
            self.logger.error(f"[Crawler] Error in crawl workflow: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "url": url
            }

