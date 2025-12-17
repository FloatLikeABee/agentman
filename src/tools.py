from langchain.tools import Tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import smtplib
import yfinance as yf
import requests
import json
import logging
from typing import Dict, List, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from .config import settings
from .models import ToolType, ToolConfig


class ToolManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tools: Dict[str, Tool] = {}
        self.tool_configs: Dict[str, ToolConfig] = {}
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