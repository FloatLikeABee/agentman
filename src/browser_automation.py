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
    
    def __init__(self, llm_provider: Optional[LLMProviderType] = None, model_name: Optional[str] = None, headless: bool = False):
        self.logger = logging.getLogger(__name__)
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.playwright = None
        self._event_loop = None  # Store event loop reference for sync tool calls
        self.headless = headless  # Whether to run browser in headless mode (False = visible browser)
        
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
        """Initialize Playwright browser - opens a REAL browser instance on your local machine"""
        if self.browser is None:
            try:
                # Start Playwright - this manages browser instances
                self.playwright = await async_playwright().start()
                
                # Launch Chromium browser - this opens a REAL browser on your local machine
                # headless=False means you'll see the browser window
                # headless=True means it runs in the background (faster, but invisible)
                self.browser = await self.playwright.chromium.launch(
                    headless=self.headless,
                    slow_mo=500 if not self.headless else 0  # Slow down actions so you can see them in visible mode
                )
                
                # Create a browser context (like an incognito window)
                self.context = await self.browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                
                # Open a new page/tab
                self.page = await self.context.new_page()
                
                mode = "visible" if not self.headless else "headless"
                self.logger.info(f"Browser initialized in {mode} mode - ready to perform actions")
            except Exception as e:
                error_msg = str(e)
                if "Executable doesn't exist" in error_msg or "BrowserType.launch" in error_msg:
                    raise RuntimeError(
                        "Playwright browsers are not installed. Please run: playwright install chromium\n"
                        "Or install all browsers: playwright install"
                    ) from e
                raise
    
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
            result = self._run_async_in_sync_context(self._navigate_async(url))
            return result
        except Exception as e:
            return f"Error navigating: {str(e)}"
    
    async def _navigate_async(self, url: str) -> str:
        """Async navigate implementation"""
        if self.browser is None:
            await self._initialize_browser()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        self.logger.info(f"Navigating to: {url}")
        try:
            await self.page.goto(url, wait_until='networkidle', timeout=30000)
            self.logger.info(f"Successfully navigated to: {url}")
            return f"Navigated to {url}"
        except Exception as e:
            error_msg = f"Error navigating to {url}: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def _click(self, selector_or_text: str) -> str:
        """Click on an element"""
        try:
            result = self._run_async_in_sync_context(self._click_async(selector_or_text))
            return result
        except Exception as e:
            return f"Error clicking: {str(e)}"
    
    async def _click_async(self, selector_or_text: str) -> str:
        """Async click implementation"""
        if self.browser is None:
            await self._initialize_browser()
        self.logger.info(f"Attempting to click: {selector_or_text}")
        try:
            # Try as CSS selector first
            await self.page.click(selector_or_text, timeout=5000)
            self.logger.info(f"Successfully clicked on {selector_or_text}")
            return f"Clicked on {selector_or_text}"
        except Exception as e1:
            # Try as text content
            try:
                await self.page.click(f"text={selector_or_text}", timeout=5000)
                self.logger.info(f"Successfully clicked on element with text: {selector_or_text}")
                return f"Clicked on element with text: {selector_or_text}"
            except Exception as e2:
                # Try as partial text match
                try:
                    await self.page.click(f"text=/{selector_or_text}/i", timeout=5000)
                    self.logger.info(f"Successfully clicked on element containing text: {selector_or_text}")
                    return f"Clicked on element containing text: {selector_or_text}"
                except Exception as e3:
                    error_msg = f"Failed to click on {selector_or_text}. Errors: {str(e1)}, {str(e2)}, {str(e3)}"
                    self.logger.error(error_msg)
                    return f"Error: {error_msg}"
    
    def _type_text(self, input_json: str) -> str:
        """Type text into an input field"""
        try:
            result = self._run_async_in_sync_context(self._type_text_async(input_json))
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
            result = self._run_async_in_sync_context(self._get_text_async(selector))
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
            result = self._run_async_in_sync_context(self._get_page_content_async(mode))
            return result
        except Exception as e:
            return f"Error getting page content: {str(e)}"
    
    async def _get_page_content_async(self, mode: str = "") -> str:
        """Async get page content implementation"""
        if self.browser is None:
            await self._initialize_browser()
        self.logger.info(f"Getting page content (mode: {mode})")
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
                result = f"Page Title: {title}\nURL: {url}\nHeadings: {', '.join(headings)}"
                self.logger.info(f"Page summary retrieved: {title}")
                return result
            else:
                # Get full text content
                text = await self.page.evaluate("() => document.body.innerText")
                # Limit to first 5000 characters to avoid token limits
                result = text[:5000] + ("..." if len(text) > 5000 else "")
                self.logger.info(f"Page content retrieved: {len(text)} characters")
                return result
        except Exception as e:
            error_msg = f"Error getting page content: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def _take_screenshot(self, filename: str = "") -> str:
        """Take a screenshot"""
        try:
            result = self._run_async_in_sync_context(self._take_screenshot_async(filename))
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
            result = self._run_async_in_sync_context(self._wait_async(input_str))
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
            result = self._run_async_in_sync_context(self._scroll_async(direction))
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
            result = self._run_async_in_sync_context(self._select_option_async(input_json))
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
            result = self._run_async_in_sync_context(self._get_url_async())
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
            # Run async execution (includes cleanup at the end)
            result = asyncio.run(self._execute_with_cleanup(instructions, max_steps))
            return result
        except Exception as e:
            self.logger.error(f"Error in browser automation execution: {e}")
            # Ensure cleanup even on error
            try:
                asyncio.run(self._cleanup_browser())
            except Exception as cleanup_error:
                self.logger.warning(f"Error during cleanup: {cleanup_error}")
            return f"Error: {str(e)}"
    
    async def _execute_with_cleanup(self, instructions: str, max_steps: int = 20) -> str:
        """Async execution with guaranteed cleanup"""
        try:
            return await self._execute_async(instructions, max_steps)
        finally:
            # Always cleanup browser resources
            await self._cleanup_browser()
    
    def _run_async_in_sync_context(self, coro):
        """Helper to run async code from sync context, handling both cases"""
        try:
            # Try to get the running event loop
            loop = asyncio.get_running_loop()
            # We're in an async context - schedule the coroutine on the existing loop
            # This ensures browser operations run in the same event loop where browser was initialized
            import concurrent.futures
            
            # Use run_coroutine_threadsafe to schedule on the existing event loop
            # This is thread-safe and allows sync code to await async operations
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            try:
                return future.result(timeout=120)  # 2 minute timeout for browser operations
            except concurrent.futures.TimeoutError:
                future.cancel()
                return "Error: Operation timed out after 2 minutes"
        except RuntimeError:
            # No running event loop - safe to use asyncio.run()
            return asyncio.run(coro)
        except Exception as e:
            self.logger.error(f"Error running async in sync context: {e}")
            return f"Error: {str(e)}"
    
    async def _execute_async(self, instructions: str, max_steps: int = 20) -> str:
        """Async execution implementation"""
        try:
            # Store the event loop for sync tool calls
            try:
                self._event_loop = asyncio.get_running_loop()
            except RuntimeError:
                self._event_loop = None
            
            await self._initialize_browser()
            
            # Wrap LLM caller in LangChain wrapper
            from .llm_langchain_wrapper import LangChainLLMWrapper
            llm_wrapper = LangChainLLMWrapper(llm_caller=self.llm)
            
            # Create ReAct agent with proper prompt template
            from langchain.agents import AgentExecutor, create_react_agent
            
            # Use custom prompt with all required variables for ReAct agent
            # Note: create_react_agent requires 'tools', 'tool_names', 'input', and 'agent_scratchpad'
            prompt = PromptTemplate(
                template="""You are a browser automation agent. Your task is to follow the user's instructions by controlling a web browser.

You have access to the following tools:

{tools}

Tool names: {tool_names}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

User instructions: {input}

IMPORTANT WORKFLOW:
1. FIRST: Understand the user's instructions and break them down into clear, sequential tasks
2. SECOND: If a URL is mentioned, use the navigate tool to go to that page first
3. THIRD: Use get_page_content to understand what's on the current page
4. FOURTH: Execute each task step by step, using the appropriate tools
5. FIFTH: Verify your actions worked by checking the page content or taking screenshots

Follow these guidelines:
- Always start by understanding the current page state with get_page_content
- Break down complex tasks into smaller, manageable steps
- Use appropriate selectors (CSS selectors, text content, or XPath)
- Wait for elements to load before interacting with them (use the wait tool if needed)
- Take screenshots if needed to verify actions
- Report what you're doing at each step in your thoughts
- If you encounter an error, try alternative approaches (different selectors, waiting longer, etc.)

Begin by analyzing the user's instructions, then navigate to the required page (if needed), understand the page, and execute the tasks step by step.

{agent_scratchpad}""",
                input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
            )
            
            agent = create_react_agent(llm_wrapper, self.browser_tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.browser_tools,
                verbose=True,
                max_iterations=max_steps,
                handle_parsing_errors=True,
                return_intermediate_steps=True  # Return steps for visibility
            )
            
            # Log the start of execution
            self.logger.info(f"Starting browser automation with instructions: {instructions[:100]}...")
            
            # Execute agent - this will:
            # 1. Have the LLM break down the instructions into tasks
            # 2. Execute each task using the browser tools
            # 3. Return the final result
            result = agent_executor.invoke({
                "input": instructions
            })
            
            # Get final page state
            final_url = await self._get_url_async()
            final_content = await self._get_page_content_async("summary")
            
            output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Build a summary of what was done
            steps_summary = ""
            if intermediate_steps:
                steps_summary = "\n\nTask Breakdown & Execution Steps:\n" + "="*60 + "\n"
                for i, step in enumerate(intermediate_steps, 1):
                    if isinstance(step, tuple) and len(step) >= 2:
                        action = step[0]
                        observation = step[1]
                        
                        # Extract tool name and input
                        tool_name = "Unknown"
                        tool_input = ""
                        if hasattr(action, 'tool'):
                            tool_name = action.tool
                            tool_input = str(action.tool_input) if hasattr(action, 'tool_input') else ""
                        elif isinstance(action, dict):
                            tool_name = action.get('tool', 'Unknown')
                            tool_input = str(action.get('tool_input', ''))
                        
                        # Format the step
                        steps_summary += f"\nStep {i}: {tool_name}\n"
                        if tool_input:
                            steps_summary += f"  Input: {tool_input[:100]}{'...' if len(tool_input) > 100 else ''}\n"
                        if observation:
                            obs_str = str(observation)[:200]
                            steps_summary += f"  Result: {obs_str}{'...' if len(str(observation)) > 200 else ''}\n"
                    else:
                        steps_summary += f"\nStep {i}: {str(step)[:150]}...\n"
                steps_summary += "\n" + "="*60 + "\n"
            
            self.logger.info(f"Browser automation completed. Final URL: {final_url}")
            
            return f"""Browser automation completed successfully.

Final URL: {final_url}

Agent Output:
{output}
{steps_summary}

Page Summary:
{final_content}"""
            
        except Exception as e:
            self.logger.error(f"Error executing browser automation: {e}")
            import traceback
            return f"Browser automation failed: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
