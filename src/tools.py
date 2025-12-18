from langchain.tools import Tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchTool
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
from urllib.parse import urljoin, urlparse
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
        # Web Search Tool
        search_tool = DuckDuckGoSearchTool()
        self.register_tool(
            "web_search",
            search_tool,
            ToolConfig(
                name="Web Search",
                tool_type=ToolType.WEB_SEARCH,
                description="Search the internet for current information",
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

        # Financial Data Tool
        financial_tool = Tool(
            name="Financial Data",
            func=self._get_financial_data,
            description="Get financial data for stocks and other financial instruments"
        )
        self.register_tool(
            "financial",
            financial_tool,
            ToolConfig(
                name="Financial Data",
                tool_type=ToolType.FINANCIAL,
                description="Get financial data for stocks and other financial instruments",
                config={
                    "alpha_vantage_key": settings.alpha_vantage_api_key
                }
            )
        )

        # Crawler Tool
        crawler_tool = Tool(
            name="Web Crawler",
            func=self._crawler,
            description="Crawl a website, extract and organize data using AI, then add to RAG collection. Input format: 'url:https://example.com,collection_name:my_collection,description:Collection description'"
        )
        self.register_tool(
            "crawler",
            crawler_tool,
            ToolConfig(
                name="Web Crawler",
                tool_type=ToolType.CRAWLER,
                description="Crawl websites, extract data, organize with AI, and add to RAG collections",
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
        """Get a tool by ID"""
        return self.tools.get(tool_id)

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

    def _get_financial_data(self, query: str) -> str:
        """Financial data tool function"""
        try:
            # Parse query for stock symbol
            if 'stock' in query.lower() or 'price' in query.lower():
                # Extract stock symbol (simple approach)
                words = query.split()
                for word in words:
                    if len(word) <= 5 and word.isupper():
                        symbol = word
                        break
                else:
                    return "Error: Please specify a stock symbol"

                # Get stock data using yfinance
                stock = yf.Ticker(symbol)
                info = stock.info
                
                if info:
                    return f"Stock: {symbol}\nPrice: ${info.get('currentPrice', 'N/A')}\nMarket Cap: ${info.get('marketCap', 'N/A'):,}\nVolume: {info.get('volume', 'N/A'):,}"
                else:
                    return f"Error: Could not fetch data for {symbol}"

            elif 'alpha_vantage' in query.lower() and settings.alpha_vantage_api_key:
                # Use Alpha Vantage API for more detailed data
                # This is a placeholder - implement based on specific needs
                return "Alpha Vantage integration available but not implemented"

            else:
                return "Please specify what financial data you need (stock price, etc.)"

        except Exception as e:
            return f"Error getting financial data: {str(e)}"

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

    def _crawler(self, input_data: str) -> str:
        """Crawler tool: Crawl website, extract data, organize with AI, add to RAG"""
        try:
            # Parse input: "url:https://example.com,collection_name:my_collection,description:Collection description"
            params = {}
            for part in input_data.split(','):
                if ':' in part:
                    key, value = part.split(':', 1)
                    params[key.strip()] = value.strip()
            
            url = params.get('url', '')
            collection_name = params.get('collection_name', '')
            description = params.get('description', '')
            
            if not url:
                return "Error: URL is required. Format: 'url:https://example.com,collection_name:name,description:desc'"
            if not collection_name:
                return "Error: collection_name is required"
            
            # Step 1: Crawl the website
            self.logger.info(f"[Crawler] Step 1/4: Crawling URL: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                self.logger.info(f"[Crawler] Step 1/4: Successfully fetched {len(response.content)} bytes")
            except requests.Timeout:
                return f"Error: Request to {url} timed out after 30 seconds"
            except requests.RequestException as e:
                return f"Error crawling URL: {str(e)}"
            
            # Step 2: Parse HTML and extract text
            self.logger.info(f"[Crawler] Step 2/4: Parsing HTML content...")
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text_content = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Also extract links and other useful data
            links = [a.get('href', '') for a in soup.find_all('a', href=True)]
            titles = [tag.get_text().strip() for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
            
            raw_data = {
                "url": url,
                "title": soup.title.string if soup.title else "",
                "headings": titles[:20],  # Limit to first 20 headings
                "links": links[:50],  # Limit to first 50 links
                "content": text_content[:50000]  # Limit content size
            }
            
            # Step 3: Use AI to organize and filter data
            if not self.rag_system:
                return "Error: RAG system not available. Cannot add data to collection."
            
            self.logger.info(f"[Crawler] Step 3/4: Preparing AI processing...")
            
            # Create LLM caller for AI processing
            try:
                # Determine provider from settings (case-insensitive)
                provider_str = settings.default_llm_provider.lower().strip()
                self.logger.info(f"Tool requested provider: '{settings.default_llm_provider}' (normalized: '{provider_str}')")
                
                if provider_str == "gemini":
                    provider = LLMProvider.GEMINI
                    api_key = settings.gemini_api_key
                    model = settings.gemini_default_model
                    self.logger.info("Selected Gemini provider for crawler tool")
                elif provider_str == "qwen":
                    provider = LLMProvider.QWEN
                    api_key = settings.qwen_api_key
                    model = settings.qwen_default_model
                    self.logger.info("Selected Qwen provider for crawler tool")
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
                
                # Create prompt for AI to organize data (limit content size to avoid timeouts)
                # Limit raw data to 5000 chars to keep prompt manageable
                raw_data_str = json.dumps(raw_data, indent=2)[:5000]
                
                organize_prompt = f"""Analyze and organize the following website data. Extract useful information, remove noise and irrelevant content, and structure it in a clear JSON format.

Website URL: {url}
Raw Data:
{raw_data_str}

Please:
1. Identify the main topics and useful information
2. Remove navigation elements, ads, and other noise
3. Organize the content into structured JSON format
4. Keep only relevant and valuable information
5. Create a clean, organized structure

Return ONLY valid JSON with this structure:
{{
    "title": "Page title",
    "summary": "Brief summary of the page content",
    "main_topics": ["topic1", "topic2", ...],
    "key_points": ["point1", "point2", ...],
    "useful_content": "Organized useful content from the page",
    "metadata": {{
        "source_url": "{url}",
        "extracted_at": "timestamp"
    }}
}}"""
                
                # Call AI to organize data with timeout handling
                self.logger.info(f"[Crawler] Step 3/4: Calling AI to organize data (this may take 30-60 seconds)...")
                try:
                    # Use timeout from settings, but cap at 60 seconds for tool execution
                    import threading
                    import time
                    
                    result_container = {'data': None, 'error': None, 'completed': False}
                    
                    def call_llm():
                        try:
                            start_time = time.time()
                            result_container['data'] = llm_caller.generate(organize_prompt)
                            elapsed = time.time() - start_time
                            self.logger.info(f"[Crawler] AI processing completed in {elapsed:.2f} seconds")
                            result_container['completed'] = True
                        except Exception as e:
                            result_container['error'] = e
                            result_container['completed'] = True
                    
                    thread = threading.Thread(target=call_llm)
                    thread.daemon = True
                    thread.start()
                    
                    # Wait with timeout (max 60 seconds for tool execution)
                    max_timeout = min(settings.api_timeout, 60)
                    thread.join(timeout=max_timeout)
                    
                    if not result_container['completed']:
                        self.logger.error(f"[Crawler] LLM call timed out after {max_timeout} seconds")
                        return f"Error: AI processing timed out after {max_timeout} seconds. The website content may be too large. Try a smaller page or increase timeout settings."
                    
                    if result_container['error']:
                        raise result_container['error']
                    
                    organized_data_str = result_container['data']
                    if not organized_data_str:
                        return "Error: AI returned empty response"
                        
                except Exception as e:
                    self.logger.error(f"[Crawler] Error calling AI: {e}")
                    return f"Error calling AI to organize data: {str(e)}"
                
                # Try to extract JSON from response
                try:
                    # Find JSON in the response (might have markdown code blocks)
                    if '```json' in organized_data_str:
                        organized_data_str = organized_data_str.split('```json')[1].split('```')[0].strip()
                    elif '```' in organized_data_str:
                        organized_data_str = organized_data_str.split('```')[1].split('```')[0].strip()
                    
                    organized_data = json.loads(organized_data_str)
                except json.JSONDecodeError:
                    # If JSON parsing fails, use the AI response as content
                    self.logger.warning("Failed to parse AI response as JSON, using as plain text")
                    organized_data = {
                        "title": raw_data.get("title", ""),
                        "summary": "AI-organized content",
                        "content": organized_data_str,
                        "metadata": {"source_url": url}
                    }
                
                # Step 4: Add organized data to RAG collection
                self.logger.info(f"[Crawler] Step 4/4: Adding data to RAG collection '{collection_name}'...")
                rag_input = RAGDataInput(
                    name=f"crawled_{collection_name}",
                    description=description or f"Data crawled from {url}",
                    format=DataFormat.JSON,
                    content=json.dumps(organized_data, indent=2),
                    tags=["crawled", "web", urlparse(url).netloc],
                    metadata={"source_url": url, "crawled_at": datetime.now().isoformat()}
                )
                
                success = self.rag_system.add_data_to_collection(collection_name, rag_input)
                
                if success:
                    key_points_count = len(organized_data.get('key_points', []))
                    self.logger.info(f"[Crawler] Successfully completed all steps!")
                    return f"Successfully crawled {url} and added organized data to RAG collection '{collection_name}'. Extracted {key_points_count} key points."
                else:
                    return f"Error: Failed to add data to RAG collection '{collection_name}'"
                    
            except Exception as e:
                self.logger.error(f"Error in AI processing or RAG storage: {e}")
                return f"Error processing data with AI or adding to RAG: {str(e)}"
                
        except requests.RequestException as e:
            return f"Error crawling URL: {str(e)}"
        except Exception as e:
            self.logger.error(f"Error in crawler tool: {e}")
            return f"Error in crawler: {str(e)}"

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