from langchain.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import smtplib
import yfinance as yf
import requests
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, quote_plus
import time
import random
from .config import settings
from .models import ToolType, ToolConfig, RAGDataInput, DataFormat, LLMProviderType
from .llm_factory import LLMFactory, LLMProvider


class ToolManager:
    def __init__(self, rag_system=None):
        self.logger = logging.getLogger(__name__)
        self.tools: Dict[str, Tool] = {}
        self.tool_configs: Dict[str, ToolConfig] = {}
        self.rag_system = rag_system
        self._initialize_default_tools()

    def _initialize_default_tools(self):
        """Initialize default tools"""
        # Web Search Tool - Custom implementation with multiple free engines
        search_tool = Tool(
            name="Web Search",
            func=self._web_search,
            description="Search the internet for current information using multiple free search engines. No API key required, unlimited usage."
        )
        self.register_tool(
            "web_search",
            search_tool,
            ToolConfig(
                name="Web Search",
                tool_type=ToolType.WEB_SEARCH,
                description="Search the internet for current information using multiple free search engines. No API key required, unlimited usage.",
                config={}
            )
        )

        # Wikipedia Tool
        wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        self.register_tool(
            "wikipedia",
            wiki_tool,
            ToolConfig(
                name="Wikipedia Search",
                tool_type=ToolType.WIKIPEDIA,
                description="Search Wikipedia for factual information",
                config={}
            )
        )

        # Calculator Tool
        calculator_tool = Tool(
            name="Calculator",
            func=self._calculator,
            description="Perform mathematical calculations"
        )
        self.register_tool(
            "calculator",
            calculator_tool,
            ToolConfig(
                name="Calculator",
                tool_type=ToolType.CALCULATOR,
                description="Perform mathematical calculations",
                config={}
            )
        )

        # Email Tool
        email_tool = Tool(
            name="Email Sender",
            func=self._send_email,
            description="Send emails using configured SMTP settings"
        )
        self.register_tool(
            "email",
            email_tool,
            ToolConfig(
                name="Email Sender",
                tool_type=ToolType.EMAIL,
                description="Send emails using configured SMTP settings",
                config={
                    "smtp_server": settings.smtp_server,
                    "smtp_port": settings.smtp_port,
                    "smtp_username": settings.smtp_username,
                    "smtp_password": settings.smtp_password
                }
            )
        )

        # Financial Data Tool (using free yfinance API)
        financial_tool = Tool(
            name="Financial Data",
            func=self._get_financial_data,
            description="Get real-time stock prices and financial data using free yfinance API. Input format: 'stock price of SYMBOL' or 'SYMBOL price' where SYMBOL is the stock ticker (e.g., AAPL, MSFT, TSLA). Returns current price, market cap, volume, and other key metrics."
        )
        self.register_tool(
            "financial",
            financial_tool,
            ToolConfig(
                name="Financial Data",
                tool_type=ToolType.FINANCIAL,
                description="Get real-time stock prices and financial data using free yfinance API. No API key required. Supports all major stock exchanges.",
                config={}
            )
        )

        # Equalizer Tool
        equalizer_tool = Tool(
            name="Decision Equalizer",
            func=self._equalizer,
            description="Use AI to help make decisions by weighing options given a scenario. Input format: 'scenario:Your decision scenario description'"
        )
        self.register_tool(
            "equalizer",
            equalizer_tool,
            ToolConfig(
                name="Decision Equalizer",
                tool_type=ToolType.EQUALIZER,
                description="Use AI to analyze scenarios and help make weighted decisions",
                config={}
            )
        )

    def register_tool(self, tool_id: str, tool: Tool, config: ToolConfig):
        """Register a new tool"""
        self.tools[tool_id] = tool
        self.tool_configs[tool_id] = config
        self.logger.info(f"Registered tool: {tool_id}")

    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get a tool by ID or name (case-insensitive)"""
        # First try exact match
        if tool_id in self.tools:
            return self.tools.get(tool_id)
        
        # Try case-insensitive match
        tool_id_lower = tool_id.lower()
        for key in self.tools.keys():
            if key.lower() == tool_id_lower:
                return self.tools.get(key)
        
        # Try matching by tool config name
        for key, config in self.tool_configs.items():
            if config.name.lower() == tool_id_lower or config.name == tool_id:
                return self.tools.get(key)
        
        return None

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        return [
            {
                'id': tool_id,
                'name': config.name,
                'tool_type': config.tool_type,
                'description': config.description,
                'is_active': config.is_active,
                'config': config.config
            }
            for tool_id, config in self.tool_configs.items()
        ]

    def update_tool_config(self, tool_id: str, config: ToolConfig) -> bool:
        """Update tool configuration"""
        try:
            if tool_id in self.tool_configs:
                self.tool_configs[tool_id] = config
                self.logger.info(f"Updated tool config: {tool_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error updating tool config {tool_id}: {e}")
            return False

    def _calculator(self, expression: str) -> str:
        """Calculator tool function"""
        try:
            # Safe evaluation of mathematical expressions
            allowed_chars = set('0123456789+-*/(). ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error in calculation: {str(e)}"

    def _send_email(self, email_data: str) -> str:
        """Email sending tool function"""
        try:
            # Parse email data (expected format: "to:email@example.com,subject:Subject,body:Email body")
            data_parts = email_data.split(',')
            email_dict = {}
            for part in data_parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    email_dict[key.strip()] = value.strip()

            to_email = email_dict.get('to', '')
            subject = email_dict.get('subject', '')
            body = email_dict.get('body', '')

            if not all([to_email, subject, body]):
                return "Error: Missing required email fields (to, subject, body)"

            # Check if SMTP is configured
            if not settings.smtp_server or not settings.smtp_username:
                return "Error: SMTP not configured"

            # Create message
            msg = MIMEMultipart()
            msg['From'] = settings.smtp_username
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            # Send email
            server = smtplib.SMTP(settings.smtp_server, settings.smtp_port)
            server.starttls()
            server.login(settings.smtp_username, settings.smtp_password)
            text = msg.as_string()
            server.sendmail(settings.smtp_username, to_email, text)
            server.quit()

            return f"Email sent successfully to {to_email}"

        except Exception as e:
            return f"Error sending email: {str(e)}"

    def _web_search(self, query: str) -> str:
        """Custom web search using multiple free search engines with no limits"""
        try:
            results = []
            query_encoded = quote_plus(query)
            
            # Try Tavily API first if API key is available (optional, has free tier)
            tavily_api_key = getattr(settings, 'tavily_api_key', None)
            if tavily_api_key:
                try:
                    self.logger.info("Trying Tavily API for web search")
                    tavily_url = "https://api.tavily.com/search"
                    payload = {
                        "api_key": tavily_api_key,
                        "query": query,
                        "search_depth": "basic",
                        "max_results": 10
                    }
                    response = requests.post(tavily_url, json=payload, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('results'):
                            for result in data['results']:
                                results.append({
                                    'title': result.get('title', 'No title'),
                                    'url': result.get('url', ''),
                                    'snippet': result.get('content', 'No description available')
                                })
                            if results:
                                self.logger.info(f"Successfully retrieved {len(results)} results from Tavily")
                                formatted_results = []
                                for i, result in enumerate(results[:10], 1):
                                    formatted_results.append(
                                        f"{i}. {result['title']}\n   URL: {result['url']}\n   {result['snippet']}"
                                    )
                                return "\n\n".join(formatted_results)
                except Exception as e:
                    self.logger.debug(f"Tavily API failed: {e}, falling back to free search engines")
            
            # Fallback to free search engines (no API key required, unlimited)
            # Try multiple free search engines as fallbacks
            search_engines = [
                {
                    "name": "Startpage",
                    "url": f"https://www.startpage.com/sp/search?query={query_encoded}",
                    "selectors": {
                        "results": "div.w-gl__result",
                        "title": "h3 a",
                        "link": "h3 a",
                        "snippet": "p.w-gl__description"
                    }
                },
                {
                    "name": "Qwant",
                    "url": f"https://www.qwant.com/?q={query_encoded}&t=web",
                    "selectors": {
                        "results": "div.web-result",
                        "title": "a",
                        "link": "a",
                        "snippet": "p.web-result-description"
                    }
                },
                {
                    "name": "Ecosia",
                    "url": f"https://www.ecosia.org/search?q={query_encoded}",
                    "selectors": {
                        "results": "div.result",
                        "title": "h2 a",
                        "link": "h2 a",
                        "snippet": "p.result-snippet"
                    }
                }
            ]
            
            # Headers to mimic a real browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
            }
            
            # Try each search engine until we get results
            for engine in search_engines:
                try:
                    self.logger.info(f"Trying {engine['name']} for query: {query}")
                    response = requests.get(engine['url'], headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    result_elements = soup.select(engine['selectors']['results'])
                    
                    if result_elements:
                        for element in result_elements[:5]:  # Limit to top 5 results per engine
                            try:
                                title_elem = element.select_one(engine['selectors']['title'])
                                link_elem = element.select_one(engine['selectors']['link'])
                                snippet_elem = element.select_one(engine['selectors']['snippet'])
                                
                                if title_elem and link_elem:
                                    title = title_elem.get_text(strip=True)
                                    link = link_elem.get('href', '')
                                    
                                    # Handle relative URLs
                                    if link and not link.startswith('http'):
                                        link = urljoin(engine['url'], link)
                                    
                                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else "No description available"
                                    
                                    if title and link:  # Only add if we have both
                                        results.append({
                                            'title': title,
                                            'url': link,
                                            'snippet': snippet
                                        })
                            except Exception as e:
                                self.logger.debug(f"Error parsing result element: {e}")
                                continue
                        
                        if results:
                            self.logger.info(f"Successfully retrieved {len(results)} results from {engine['name']}")
                            break  # We got results, no need to try other engines
                            
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"Failed to search with {engine['name']}: {e}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Error parsing {engine['name']} results: {e}")
                    continue
            
            # If no results from scraping, try DuckDuckGo instant answer API as final fallback
            if not results:
                try:
                    self.logger.info("Trying DuckDuckGo instant answer API as final fallback")
                    ddg_url = f"https://api.duckduckgo.com/?q={query_encoded}&format=json&no_html=1&skip_disambig=1"
                    response = requests.get(ddg_url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('AbstractText'):
                            results.append({
                                'title': data.get('Heading', 'DuckDuckGo Result'),
                                'url': data.get('AbstractURL', ''),
                                'snippet': data.get('AbstractText', '')
                            })
                except Exception as e:
                    self.logger.debug(f"DuckDuckGo API fallback failed: {e}")
            
            # Format results
            if results:
                formatted_results = []
                for i, result in enumerate(results[:10], 1):  # Limit to top 10 total results
                    formatted_results.append(
                        f"{i}. {result['title']}\n   URL: {result['url']}\n   {result['snippet']}"
                    )
                return "\n\n".join(formatted_results)
            else:
                return f"No search results found for: {query}. Please try rephrasing your query."
                
        except Exception as e:
            self.logger.error(f"Error in web search: {e}")
            return f"Error performing web search: {str(e)}. Please try again."

    def _get_financial_data(self, query: str) -> str:
        """Financial data tool function using free yfinance API (no API key required)"""
        try:
            import time
            import re
            from datetime import datetime
            
            # Parse query for stock symbol - improved extraction
            symbol = None
            
            # Try to extract symbol from common patterns
            words = query.split()
            for i, word in enumerate(words):
                # Look for uppercase ticker symbols (1-5 characters)
                if len(word) <= 5 and word.isupper() and word.isalpha():
                    symbol = word
                    break
                # Look for patterns like "AAPL stock" or "stock AAPL"
                if word.lower() in ['stock', 'price', 'quote', 'of'] and i + 1 < len(words):
                    potential_symbol = words[i + 1].upper().strip('.,;:!?')
                    if len(potential_symbol) <= 5 and potential_symbol.isalpha():
                        symbol = potential_symbol
                        break
            
            if not symbol:
                # Try to find symbol in the query more flexibly using regex
                symbol_match = re.search(r'\b([A-Z]{1,5})\b', query.upper())
                if symbol_match:
                    symbol = symbol_match.group(1)
            
            if not symbol:
                return "FORMAT: Please provide a stock symbol (e.g., AAPL, MSFT, TSLA). Example: 'stock price of AAPL' or 'AAPL price'"
            
            # Get stock data using yfinance (free, no API key required)
            # Use history first as it's less likely to hit rate limits
            try:
                stock = yf.Ticker(symbol)
                current_price = None
                info = {}
                
                # Method 1: Try history first (less rate-limited than info)
                try:
                    # Try 1 day history with daily intervals (most reliable)
                    hist = stock.history(period="1d", interval="1d")
                    if not hist.empty and len(hist) > 0:
                        current_price = float(hist['Close'].iloc[-1])
                except:
                    try:
                        # Fallback: Try 5 day history
                        hist = stock.history(period="5d", interval="1d")
                        if not hist.empty and len(hist) > 0:
                            current_price = float(hist['Close'].iloc[-1])
                    except:
                        pass
                
                # Method 2: Try info if history worked (for additional data)
                # Only try info if we got price from history, to avoid rate limits
                if current_price is not None:
                    try:
                        info = stock.info
                        if info and not info.get('currentPrice'):
                            # If info doesn't have currentPrice, use the one from history
                            pass
                    except Exception as e1:
                        # If info fails due to rate limit, that's okay - we have price from history
                        self.logger.debug(f"Info unavailable for {symbol} (may be rate limited): {e1}")
                        info = {}
                
                # Method 3: If history failed, try info as last resort
                if current_price is None:
                    try:
                        info = stock.info
                        if info:
                            # Try various price fields
                            current_price = (info.get('currentPrice') or 
                                            info.get('regularMarketPrice') or 
                                            info.get('regularMarketPreviousClose') or
                                            info.get('previousClose') or
                                            info.get('ask') or
                                            info.get('bid'))
                            if current_price:
                                current_price = float(current_price)
                    except Exception as e1:
                        # Check if it's a rate limit error
                        error_str = str(e1).lower()
                        if 'rate limit' in error_str or 'too many' in error_str:
                            # Return helpful message about rate limiting
                            return f"Stock: {symbol} | Status: Rate limited by Yahoo Finance. Please wait 30-60 seconds and try again. The free API has usage limits."
                        self.logger.debug(f"Error getting info for {symbol}: {e1}")
                        info = {}
                
                # Method 4: Try fast_info as final fallback
                if current_price is None:
                    try:
                        fast_info = stock.fast_info
                        if hasattr(fast_info, 'last_price') and fast_info.last_price:
                            current_price = float(fast_info.last_price)
                        elif hasattr(fast_info, 'regular_market_price') and fast_info.regular_market_price:
                            current_price = float(fast_info.regular_market_price)
                    except:
                        pass
                
                # Build structured response that won't break agent parsing
                result_parts = [f"Stock: {symbol}"]
                
                if current_price:
                    result_parts.append(f"Price: ${current_price:.2f}")
                elif info and info.get('previousClose'):
                    # If we have previous close but no current price, show that
                    prev_close = float(info.get('previousClose'))
                    result_parts.append(f"Previous Close: ${prev_close:.2f}")
                    result_parts.append("Note: Current price unavailable (market may be closed)")
                else:
                    # Last resort: try to get any price data
                    if info:
                        for price_key in ['regularMarketPreviousClose', 'previousClose', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow']:
                            if price_key in info and info[price_key]:
                                price_val = float(info[price_key])
                                result_parts.append(f"{price_key.replace('fiftyTwo', '52W').replace('regularMarket', '')}: ${price_val:.2f}")
                                break
                    
                    if len(result_parts) == 1:  # Only has "Stock: SYMBOL"
                        result_parts.append("Price: Unable to fetch current price. Market may be closed or symbol invalid.")
                
                # Add more data if available
                if info:
                    if 'marketCap' in info and info['marketCap']:
                        try:
                            market_cap = float(info['marketCap'])
                            if market_cap >= 1e12:
                                market_cap_str = f"${market_cap/1e12:.2f}T"
                            elif market_cap >= 1e9:
                                market_cap_str = f"${market_cap/1e9:.2f}B"
                            elif market_cap >= 1e6:
                                market_cap_str = f"${market_cap/1e6:.2f}M"
                            else:
                                market_cap_str = f"${market_cap:,.0f}"
                            result_parts.append(f"Market Cap: {market_cap_str}")
                        except:
                            pass
                    
                    if 'volume' in info and info['volume']:
                        try:
                            result_parts.append(f"Volume: {int(info['volume']):,}")
                        except:
                            pass
                    
                    if current_price and 'previousClose' in info and info['previousClose']:
                        try:
                            prev_close = float(info['previousClose'])
                            change = current_price - prev_close
                            change_pct = (change / prev_close) * 100
                            result_parts.append(f"Previous Close: ${prev_close:.2f}")
                            result_parts.append(f"Change: ${change:+.2f} ({change_pct:+.2f}%)")
                        except:
                            pass
                    
                    if '52WeekHigh' in info and info['52WeekHigh']:
                        try:
                            result_parts.append(f"52W High: ${float(info['52WeekHigh']):.2f}")
                        except:
                            pass
                    
                    if '52WeekLow' in info and info['52WeekLow']:
                        try:
                            result_parts.append(f"52W Low: ${float(info['52WeekLow']):.2f}")
                        except:
                            pass
                
                return "\n".join(result_parts)
                
            except Exception as yf_error:
                # Handle yfinance-specific errors gracefully with structured format
                error_msg = str(yf_error).lower()
                if 'rate limit' in error_msg or 'too many' in error_msg or '429' in str(yf_error):
                    return f"Stock: {symbol} | Status: Rate limited. Please wait 10 seconds and try again."
                elif 'not found' in error_msg or 'invalid' in error_msg or '404' in str(yf_error):
                    return f"Stock: {symbol} | Status: Symbol not found. Please verify the stock symbol is correct."
                else:
                    # Try one more time with a simpler approach
                    try:
                        stock = yf.Ticker(symbol)
                        fast_info = stock.fast_info
                        if hasattr(fast_info, 'last_price'):
                            price = float(fast_info.last_price)
                            return f"Stock: {symbol}\nPrice: ${price:.2f}"
                    except:
                        return f"Stock: {symbol} | Status: Unable to fetch data. Please verify symbol and try again later."

        except Exception as e:
            # Return structured error format that won't break agent parsing
            return f"Status: Error retrieving financial data. Please provide a valid stock symbol (e.g., AAPL, MSFT)."

    def create_custom_tool(self, tool_id: str, name: str, description: str, func) -> bool:
        """Create a custom tool"""
        try:
            custom_tool = Tool(
                name=name,
                func=func,
                description=description
            )
            
            config = ToolConfig(
                name=name,
                tool_type=ToolType.CUSTOM,
                description=description,
                config={}
            )
            
            self.register_tool(tool_id, custom_tool, config)
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating custom tool {tool_id}: {e}")
            return False

    def _equalizer(self, input_data: str) -> str:
        """Equalizer tool: Use AI to help make decisions by weighing options"""
        try:
            # Parse input: "scenario:Your decision scenario description"
            if ':' in input_data:
                key, scenario = input_data.split(':', 1)
                scenario = scenario.strip()
            else:
                scenario = input_data.strip()
            
            if not scenario:
                return "Error: Please provide a decision scenario. Format: 'scenario:Your scenario description'"
            
            # Create LLM caller
            try:
                # Determine provider from settings (case-insensitive)
                provider_str = settings.default_llm_provider.lower().strip()
                self.logger.info(f"Tool requested provider: '{settings.default_llm_provider}' (normalized: '{provider_str}')")
                
                if provider_str == "gemini":
                    provider = LLMProvider.GEMINI
                    api_key = settings.gemini_api_key
                    model = settings.gemini_default_model
                    self.logger.info("Selected Gemini provider for equalizer tool")
                elif provider_str == "qwen":
                    provider = LLMProvider.QWEN
                    api_key = settings.qwen_api_key
                    model = settings.qwen_default_model
                    self.logger.info("Selected Qwen provider for equalizer tool")
                else:
                    # Fallback: try Qwen first, then Gemini
                    self.logger.warning(f"Unknown provider '{settings.default_llm_provider}', trying Qwen first")
                    try:
                        provider = LLMProvider.QWEN
                        api_key = settings.qwen_api_key
                        model = settings.qwen_default_model
                        llm_caller = LLMFactory.create_caller(provider=provider, api_key=api_key, model=model)
                        self.logger.info("Successfully initialized Qwen as fallback")
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize Qwen: {e}, trying Gemini")
                        provider = LLMProvider.GEMINI
                        api_key = settings.gemini_api_key
                        model = settings.gemini_default_model
                        llm_caller = LLMFactory.create_caller(provider=provider, api_key=api_key, model=model)
                        self.logger.info("Successfully initialized Gemini as fallback")
                
                if 'llm_caller' not in locals():
                    self.logger.info(f"Initializing {provider.value} with model {model}")
                    llm_caller = LLMFactory.create_caller(
                        provider=provider,
                        api_key=api_key,
                        model=model
                    )
                
                # Create decision analysis prompt
                decision_prompt = f"""You are a decision-making assistant. Analyze the following scenario and help make a well-reasoned decision by weighing the options.

Scenario:
{scenario}

Please provide:
1. Identify all possible options/choices
2. List pros and cons for each option
3. Assign weights/scores to each factor (1-10 scale)
4. Calculate weighted scores for each option
5. Provide a clear recommendation with reasoning
6. Include any risks or considerations

Format your response as structured JSON:
{{
    "options": [
        {{
            "name": "Option 1",
            "pros": ["pro1", "pro2"],
            "cons": ["con1", "con2"],
            "factors": [
                {{"factor": "Factor name", "weight": 8, "score": 7}},
                ...
            ],
            "total_score": 0,
            "weighted_score": 0
        }},
        ...
    ],
    "analysis": "Overall analysis of the decision",
    "recommendation": {{
        "option": "Recommended option",
        "reasoning": "Why this option is recommended",
        "confidence": "high/medium/low"
    }},
    "risks": ["risk1", "risk2"],
    "considerations": ["consideration1", "consideration2"]
}}

Calculate weighted scores by: sum(factor.weight * factor.score) / sum(factor.weight) for each option."""
                
                # Call AI for decision analysis
                self.logger.info("Calling AI for decision analysis...")
                response = llm_caller.generate(decision_prompt)
                
                # Try to extract JSON from response
                try:
                    # Find JSON in the response
                    if '```json' in response:
                        response = response.split('```json')[1].split('```')[0].strip()
                    elif '```' in response:
                        response = response.split('```')[1].split('```')[0].strip()
                    
                    decision_data = json.loads(response)
                    
                    # Format the response nicely
                    result = f"Decision Analysis for: {scenario}\n\n"
                    result += f"Analysis: {decision_data.get('analysis', 'N/A')}\n\n"
                    
                    result += "Options Evaluation:\n"
                    for i, option in enumerate(decision_data.get('options', []), 1):
                        result += f"\n{i}. {option.get('name', 'Option')}\n"
                        result += f"   Pros: {', '.join(option.get('pros', []))}\n"
                        result += f"   Cons: {', '.join(option.get('cons', []))}\n"
                        result += f"   Weighted Score: {option.get('weighted_score', 0):.2f}\n"
                    
                    recommendation = decision_data.get('recommendation', {})
                    result += f"\nRecommendation: {recommendation.get('option', 'N/A')}\n"
                    result += f"Reasoning: {recommendation.get('reasoning', 'N/A')}\n"
                    result += f"Confidence: {recommendation.get('confidence', 'N/A')}\n"
                    
                    if decision_data.get('risks'):
                        result += f"\nRisks: {', '.join(decision_data.get('risks', []))}\n"
                    if decision_data.get('considerations'):
                        result += f"Considerations: {', '.join(decision_data.get('considerations', []))}\n"
                    
                    return result
                    
                except json.JSONDecodeError:
                    # If JSON parsing fails, return the raw response
                    return f"Decision Analysis:\n\n{response}"
                    
            except Exception as e:
                self.logger.error(f"Error calling AI for decision analysis: {e}")
                return f"Error in AI decision analysis: {str(e)}"
                
        except Exception as e:
            self.logger.error(f"Error in equalizer tool: {e}")
            return f"Error in equalizer: {str(e)}" 