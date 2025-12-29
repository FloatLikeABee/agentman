"""
Qwen LLM Caller
Integrates with Alibaba's Qwen API via DashScope
"""
from openai import OpenAI
from typing import Dict, Any, List, Iterator
from src.llm_factory import BaseLLMCaller, LLMFactory, LLMProvider


# API Configuration - loaded from settings
from src.config import settings
QWEN_API_KEY = settings.qwen_api_key
QWEN_BASE_URL = settings.qwen_base_url


class QwenCaller(BaseLLMCaller):
    """Qwen API caller implementation using OpenAI-compatible interface"""

    def __init__(self, api_key: str = QWEN_API_KEY, model: str = "qwen3-max", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.base_url = kwargs.get("base_url", QWEN_BASE_URL)
        # Use timeout from settings, default to 120 seconds
        self.timeout = kwargs.get("timeout", settings.api_timeout)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        self.default_temperature = kwargs.get("temperature", 0.7)
        self.default_max_tokens = kwargs.get("max_tokens", 8192 * 8)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from Qwen"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat with Qwen using a list of messages"""
        try:
            # Use timeout from kwargs or instance default
            timeout = kwargs.get("timeout", self.timeout)
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.default_temperature),
                max_tokens=kwargs.get("max_tokens", self.default_max_tokens),
                stream=False,
                timeout=timeout
            )

            return completion.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                self.logger.error(f"Qwen API request timed out after {self.timeout} seconds: {e}")
                raise TimeoutError(f"Qwen API request timed out after {self.timeout} seconds")
            self.logger.error(f"Qwen API request failed: {e}")
            raise

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream responses from Qwen"""
        try:
            messages = [{"role": "user", "content": prompt}]
            # Use timeout from kwargs or instance default
            timeout = kwargs.get("timeout", self.timeout)

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.default_temperature),
                max_tokens=kwargs.get("max_tokens", self.default_max_tokens),
                stream=True,
                timeout=timeout
            )

            for chunk in completion:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                self.logger.error(f"Qwen streaming request timed out after {self.timeout} seconds: {e}")
                raise TimeoutError(f"Qwen streaming request timed out after {self.timeout} seconds")
            self.logger.error(f"Qwen streaming request failed: {e}")
            raise


# Register Qwen caller with the factory
LLMFactory.register_caller(LLMProvider.QWEN, QwenCaller)
