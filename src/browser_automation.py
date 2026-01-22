"""
Browser Automation Tool using LangChain + Playwright
Allows AI agents to control a browser and follow user instructions
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
import json
import time
from .config import settings
from .llm_factory import LLMFactory, LLMProvider
from .models import LLMProviderType


class BrowserAutomationTool:
    """Browser automation tool using Playwright and LangChain"""
    
    def __init__(self, llm_provider: Optional[LLMProviderType] = None, model_name: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.playwright = None
        
        # Setup LLM for agent
        provider_str = llm_provider.value if llm_provider else settings.default_llm_provider.lower()
        if provider_str == "gemini":
            provider = LLMProvider.GEMINI
            api_key = settings.gemini_api_key
            model = model_name or settings.gemini_default_model
        elif provider_str == "qwen":
            provider = LLMProvider.QWEN
            api_key = settings.qwen_api_key
            model = model_name or settings.qwen_default_model
        elif provider_str == "mistral":
            provider = LLMProvider.MISTRAL
            api_key = settings.mistral_api_key
            model = model_name or settings.mistral_default_model
        else:
            provider = LLMProvider.GEMINI
            api_key = settings.gemini_api_key
            model = settings.gemini_default_model
        
        self.llm = LLMFactory.create_caller(
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=0.3,  # Lower temperature for more deterministic actions
            max_tokens=4096
        )
        
        # Create browser tools
        self.browser_tools = self._create_browser_tools()
        
    def _create_browser_tools(self) -> List[Tool]:
        """Create LangChain tools for browser automation"""
        tools = [
            Tool(
                name="navigate",
                func=self._navigate,
                description="Navigate to a URL. Input: URL string (e.g., 'https://example.com')"
            ),
            Tool(
                name="click",
                func=self._click,
                description="Click on an element. Input: CSS selector or text content (e.g., 'button.submit' or 'Login')"
            ),
            Tool(
                name="type",
                func=self._type_text,
                description="Type text into an input field. Input: JSON string with 'selector' and 'text' keys (e.g., '{\"selector\": \"input[name=\\\"email\\\"]\", \"text\": \"user@example.com\"}')"
            ),
            Tool(
                name="get_text",
                func=self._get_text,
                description="Get text content from an element. Input: CSS selector (e.g., 'h1.title')"
            ),
            Tool(
                name="get_page_content",
                func=self._get_page_content,
                description="Get the full text content of the current page. Input: empty string or 'summary' for summary"
            ),
            Tool(
                name="screenshot",
                func=self._take_screenshot,
                description="Take a screenshot of the current page. Input: optional filename (e.g., 'screenshot.png')"
            ),
            Tool(
                name="wait",
                func=self._wait,
                description="Wait for a specified time or element. Input: number of seconds or CSS selector (e.g., '5' or '.loading-complete')"
            ),
            Tool(
                name="scroll",
                func=self._scroll,
                description="Scroll the page. Input: 'up', 'down', 'top', 'bottom', or number of pixels (e.g., 'down' or '500')"
            ),
            Tool(
                name="select_option",
                func=self._select_option,
                description="Select an option from a dropdown. Input: JSON string with 'selector' and 'value' keys (e.g., '{\"selector\": \"select#country\", \"value\": \"US\"}')"
            ),
            Tool(
                name="get_url",
                func=self._get_url,
                description="Get the current page URL. Input: empty string"
            ),
        ]
        return tools
    
    async def _initialize_browser(self):
        """Initialize Playwright browser"""
        if self.browser is None:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            self.page = await self.context.new_page()
            self.logger.info("Browser initialized")
    
    async def _cleanup_browser(self):
        """Clean up browser resources"""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            self.browser = None
            self.context = None
            self.page = None
            self.playwright = None
            self.logger.info("Browser cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up browser: {e}")
    
    def _navigate(self, url: str) -> str:
        """Navigate to a URL"""
        try:
            result = asyncio.run(self._navigate_async(url))
            return result
        except Exception as e:
            return f"Error navigating: {str(e)}"
    
    async def _navigate_async(self, url: str) -> str:
        """Async navigate implementation"""
        await self._initialize_browser()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        await self.page.goto(url, wait_until='networkidle', timeout=30000)
        return f"Navigated to {url}"
    
    def _click(self, selector_or_text: str) -> str:
        """Click on an element"""
        try:
            result = asyncio.run(self._click_async(selector_or_text))
            return result
        except Exception as e:
            return f"Error clicking: {str(e)}"
    
    async def _click_async(self, selector_or_text: str) -> str:
        """Async click implementation"""
        await self._initialize_browser()
        try:
            # Try as CSS selector first
            await self.page.click(selector_or_text, timeout=5000)
            return f"Clicked on {selector_or_text}"
        except:
            # Try as text content
            try:
                await self.page.click(f"text={selector_or_text}", timeout=5000)
                return f"Clicked on element with text: {selector_or_text}"
            except:
                # Try as partial text match
                await self.page.click(f"text=/{selector_or_text}/i", timeout=5000)
                return f"Clicked on element containing text: {selector_or_text}"
    
    def _type_text(self, input_json: str) -> str:
        """Type text into an input field"""
        try:
            result = asyncio.run(self._type_text_async(input_json))
            return result
        except Exception as e:
            return f"Error typing text: {str(e)}"
    
    async def _type_text_async(self, input_json: str) -> str:
        """Async type text implementation"""
        await self._initialize_browser()
        try:
            data = json.loads(input_json)
            selector = data.get('selector', '')
            text = data.get('text', '')
            await self.page.fill(selector, text)
            return f"Typed '{text}' into {selector}"
        except json.JSONDecodeError:
            # If not JSON, assume it's just a selector and we'll type into it
            await self.page.fill(input_json, '')
            return f"Cleared and ready to type in {input_json}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _get_text(self, selector: str) -> str:
        """Get text content from an element"""
        try:
            result = asyncio.run(self._get_text_async(selector))
            return result
        except Exception as e:
            return f"Error getting text: {str(e)}"
    
    async def _get_text_async(self, selector: str) -> str:
        """Async get text implementation"""
        await self._initialize_browser()
        try:
            text = await self.page.text_content(selector)
            return text or f"No text found for selector: {selector}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _get_page_content(self, mode: str = "") -> str:
        """Get page content"""
        try:
            result = asyncio.run(self._get_page_content_async(mode))
            return result
        except Exception as e:
            return f"Error getting page content: {str(e)}"
    
    async def _get_page_content_async(self, mode: str = "") -> str:
        """Async get page content implementation"""
        await self._initialize_browser()
        try:
            if mode == "summary":
                # Get a summary of the page
                title = await self.page.title()
                url = self.page.url
                headings = await self.page.evaluate("""
                    () => {
                        const headings = Array.from(document.querySelectorAll('h1, h2, h3'));
                        return headings.map(h => h.textContent.trim()).slice(0, 10);
                    }
                """)
                return f"Page Title: {title}\nURL: {url}\nHeadings: {', '.join(headings)}"
            else:
                # Get full text content
                text = await self.page.evaluate("() => document.body.innerText")
                # Limit to first 5000 characters to avoid token limits
                return text[:5000] + ("..." if len(text) > 5000 else "")
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _take_screenshot(self, filename: str = "") -> str:
        """Take a screenshot"""
        try:
            result = asyncio.run(self._take_screenshot_async(filename))
            return result
        except Exception as e:
            return f"Error taking screenshot: {str(e)}"
    
    async def _take_screenshot_async(self, filename: str = "") -> str:
        """Async screenshot implementation"""
        await self._initialize_browser()
        if not filename:
            filename = f"screenshot_{int(time.time())}.png"
        await self.page.screenshot(path=filename)
        return f"Screenshot saved to {filename}"
    
    def _wait(self, input_str: str) -> str:
        """Wait for time or element"""
        try:
            result = asyncio.run(self._wait_async(input_str))
            return result
        except Exception as e:
            return f"Error waiting: {str(e)}"
    
    async def _wait_async(self, input_str: str) -> str:
        """Async wait implementation"""
        await self._initialize_browser()
        try:
            # Try to parse as number (seconds)
            seconds = float(input_str)
            await asyncio.sleep(seconds)
            return f"Waited {seconds} seconds"
        except ValueError:
            # Assume it's a CSS selector
            await self.page.wait_for_selector(input_str, timeout=10000)
            return f"Waited for element: {input_str}"
    
    def _scroll(self, direction: str) -> str:
        """Scroll the page"""
        try:
            result = asyncio.run(self._scroll_async(direction))
            return result
        except Exception as e:
            return f"Error scrolling: {str(e)}"
    
    async def _scroll_async(self, direction: str) -> str:
        """Async scroll implementation"""
        await self._initialize_browser()
        try:
            if direction.lower() == 'down':
                await self.page.evaluate("window.scrollBy(0, 500)")
            elif direction.lower() == 'up':
                await self.page.evaluate("window.scrollBy(0, -500)")
            elif direction.lower() == 'top':
                await self.page.evaluate("window.scrollTo(0, 0)")
            elif direction.lower() == 'bottom':
                await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            else:
                # Try to parse as pixels
                pixels = int(direction)
                await self.page.evaluate(f"window.scrollBy(0, {pixels})")
            return f"Scrolled {direction}"
        except ValueError:
            return f"Invalid scroll direction: {direction}. Use 'up', 'down', 'top', 'bottom', or a number of pixels"
    
    def _select_option(self, input_json: str) -> str:
        """Select an option from a dropdown"""
        try:
            result = asyncio.run(self._select_option_async(input_json))
            return result
        except Exception as e:
            return f"Error selecting option: {str(e)}"
    
    async def _select_option_async(self, input_json: str) -> str:
        """Async select option implementation"""
        await self._initialize_browser()
        try:
            data = json.loads(input_json)
            selector = data.get('selector', '')
            value = data.get('value', '')
            await self.page.select_option(selector, value)
            return f"Selected '{value}' in {selector}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _get_url(self, _: str = "") -> str:
        """Get current URL"""
        try:
            result = asyncio.run(self._get_url_async())
            return result
        except Exception as e:
            return f"Error getting URL: {str(e)}"
    
    async def _get_url_async(self) -> str:
        """Async get URL implementation"""
        await self._initialize_browser()
        return self.page.url
    
    def execute(self, instructions: str, max_steps: int = 20) -> str:
        """
        Execute browser automation instructions using LangChain agent
        
        Args:
            instructions: Natural language instructions for the browser
            max_steps: Maximum number of agent steps to take
            
        Returns:
            String with execution results
        """
        try:
            # Run async execution
            result = asyncio.run(self._execute_async(instructions, max_steps))
            return result
        except Exception as e:
            self.logger.error(f"Error in browser automation execution: {e}")
            return f"Error: {str(e)}"
        finally:
            # Ensure cleanup
            try:
                asyncio.run(self._cleanup_browser())
            except:
                pass
    
    async def _execute_async(self, instructions: str, max_steps: int = 20) -> str:
        """Async execution implementation"""
        try:
            await self._initialize_browser()
            
            # Create agent prompt
            prompt = PromptTemplate.from_template("""
You are a browser automation agent. Your task is to follow the user's instructions by controlling a web browser.

Available tools:
{tools}

User instructions: {instructions}

Follow these guidelines:
1. Break down complex tasks into smaller steps
2. Use the get_page_content tool to understand the current page state
3. Use appropriate selectors (CSS selectors, text content, or XPath)
4. Wait for elements to load before interacting with them
5. Take screenshots if needed to verify actions
6. Report what you're doing at each step

Begin by understanding the current page, then execute the instructions step by step.

{agent_scratchpad}
""")
            
            # Wrap LLM caller in LangChain wrapper
            from .llm_langchain_wrapper import LangChainLLMWrapper
            llm_wrapper = LangChainLLMWrapper(llm_caller=self.llm)
            
            # Create ReAct agent
            from langchain.agents import AgentExecutor, create_react_agent
            
            agent = create_react_agent(llm_wrapper, self.browser_tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.browser_tools,
                verbose=True,
                max_iterations=max_steps,
                handle_parsing_errors=True
            )
            
            # Build tools description
            tools_desc = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.browser_tools])
            
            # Execute agent (synchronous invoke since LangChain wrapper handles it)
            result = agent_executor.invoke({
                "instructions": instructions,
                "tools": tools_desc
            })
            
            # Get final page state
            final_url = await self._get_url_async()
            final_content = await self._get_page_content_async("summary")
            
            output = result.get("output", "")
            
            return f"""Browser automation completed successfully.

Final URL: {final_url}

Agent Output:
{output}

Page Summary:
{final_content}"""
            
        except Exception as e:
            self.logger.error(f"Error executing browser automation: {e}")
            import traceback
            return f"Browser automation failed: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
