"""
Gemini LLM Caller
Integrates with Google's Gemini API
"""
import google.generativeai as genai
from typing import Dict, Any, List, Iterator
from src.llm_factory import BaseLLMCaller, LLMFactory, LLMProvider


# API Configuration - loaded from settings
from src.config import settings
GEMINI_API_KEY = settings.gemini_api_key


class GeminiCaller(BaseLLMCaller):
    """Gemini API caller implementation"""

    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = "gemini-2.5-flash", **kwargs):
        super().__init__(api_key, model, **kwargs)
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.default_temperature = kwargs.get("temperature", 0.7)
        self.default_max_tokens = kwargs.get("max_tokens", 8192 * 8)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from Gemini"""
        try:
            response = self.model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", self.default_temperature),
                    max_output_tokens=kwargs.get("max_tokens", self.default_max_tokens)
                )
            )

            # Handle different response types
            if isinstance(response, str):
                return response
            elif hasattr(response, 'text'):
                return response.text
            else:
                # Try to extract text from response object
                return str(response)

        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error messages
            if "DNS resolution failed" in error_msg or "503" in error_msg:
                self.logger.error(f"Gemini API DNS/network error: {e}")
                raise ConnectionError(
                    f"Failed to connect to Gemini API. This may be due to:\n"
                    f"1. Network connectivity issues\n"
                    f"2. DNS resolution problems\n"
                    f"3. Firewall/proxy blocking the connection\n"
                    f"4. Gemini API service temporarily unavailable\n\n"
                    f"Original error: {error_msg}"
                )
            elif "timeout" in error_msg.lower() or "Timeout" in error_msg:
                self.logger.error(f"Gemini API timeout: {e}")
                raise TimeoutError(
                    f"Request to Gemini API timed out. The API may be slow or unavailable.\n"
                    f"Original error: {error_msg}"
                )
            else:
                self.logger.error(f"Gemini API request failed: {e}")
                raise

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat with Gemini using a list of messages"""
        try:
            # Convert messages to Gemini format
            gemini_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                gemini_messages.append({"role": role, "parts": [content]})

            # Generate response
            response = self.model.generate_content(
                contents=gemini_messages,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", self.default_temperature),
                    max_output_tokens=kwargs.get("max_tokens", self.default_max_tokens)
                )
            )

            # Handle different response types
            if isinstance(response, str):
                return response
            elif hasattr(response, 'text'):
                return response.text
            else:
                # Try to extract text from response object
                return str(response)

        except Exception as e:
            self.logger.error(f"Gemini chat request failed: {e}")
            raise

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream responses from Gemini"""
        try:
            response = self.model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", self.default_temperature),
                    max_output_tokens=kwargs.get("max_tokens", self.default_max_tokens)
                ),
                stream=True
            )

            # Gemini streaming works by iterating over the response chunks
            for chunk in response:
                if isinstance(chunk, str):
                    yield chunk
                elif hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
                else:
                    # Try to extract text from chunk object
                    chunk_text = str(chunk)
                    if chunk_text:
                        yield chunk_text

        except Exception as e:
            self.logger.error(f"Gemini streaming request failed: {e}")
            raise


# Register Gemini caller with the factory
LLMFactory.register_caller(LLMProvider.GEMINI, GeminiCaller)
