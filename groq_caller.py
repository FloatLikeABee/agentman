"""
Groq LLM Caller
Integrates with Groq's API using the official `groq` Python client.
"""
from typing import Dict, Any, List, Iterator

from groq import Groq  # type: ignore[import]

from src.llm_factory import BaseLLMCaller, LLMFactory, LLMProvider
from src.config import settings


GROQ_API_KEY = getattr(settings, "groq_api_key", "") or ""


class GroqCaller(BaseLLMCaller):
    """Groq API caller implementation using groq.Client."""

    def __init__(self, api_key: str = GROQ_API_KEY, model: str = "llama-3.3-70b-versatile", **kwargs):
        # NOTE: `model` is any valid Groq chat model, for example:
        # - "llama-3.3-70b-versatile"
        # - "llama-3.1-70b-versatile"
        # - "llama-3.2-90b-vision-preview"
        super().__init__(api_key, model, **kwargs)
        self.client = Groq(api_key=self.api_key)
        self.default_temperature = kwargs.get("temperature", 0.7)
        # Groq supports generous context sizes; keep our default similar to others
        self.default_max_tokens = kwargs.get("max_tokens", 8192)

    def _get_params(self, **overrides: Any) -> Dict[str, Any]:
        """Merge default generation parameters with overrides."""
        return {
            "temperature": overrides.get("temperature", self.default_temperature),
            "max_tokens": overrides.get("max_tokens", self.default_max_tokens),
        }

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from Groq using a single user message."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat with Groq using a list of messages."""
        try:
            params = self._get_params(**kwargs)
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=params["temperature"],
                max_tokens=params["max_tokens"],
                stream=False,
            )
            if (
                hasattr(completion, "choices")
                and completion.choices
                and completion.choices[0].message
                and getattr(completion.choices[0].message, "content", None)
            ):
                return completion.choices[0].message.content or ""
            return str(completion)
        except Exception as e:  # pragma: no cover - network / API layer
            self.logger.error(f"Groq chat request failed: {e}")
            raise

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream responses from Groq."""
        try:
            params = self._get_params(**kwargs)
            messages = [{"role": "user", "content": prompt}]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=params["temperature"],
                max_tokens=params["max_tokens"],
                stream=True,
            )
            for chunk in completion:
                try:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        yield content
                except Exception:
                    # If the response shape is unexpected, fall back to stringifying
                    text = str(chunk)
                    if text:
                        yield text
        except Exception as e:  # pragma: no cover - network / API layer
            self.logger.error(f"Groq streaming request failed: {e}")
            raise


# Register Groq caller with the factory
LLMFactory.register_caller(LLMProvider.GROQ, GroqCaller)

