"""
Mistral LLM Caller
Integrates with Mistral AI API
"""
from mistralai import Mistral
from typing import Dict, Any, List, Iterator
from src.llm_factory import BaseLLMCaller, LLMFactory, LLMProvider


# API Configuration - loaded from settings
from src.config import settings
MISTRAL_API_KEY = settings.mistral_api_key


class MistralCaller(BaseLLMCaller):
    """Mistral AI API caller implementation"""

    def __init__(self, api_key: str = MISTRAL_API_KEY, model: str = "mistral-large-latest", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.client = Mistral(api_key=self.api_key)
        self.default_temperature = kwargs.get("temperature", 0.7)
        self.default_max_tokens = kwargs.get("max_tokens", 8192 * 8)
        self.agent_id = kwargs.get("agent_id", None)  # Optional agent ID for agent-based completions

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from Mistral"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat with Mistral using a list of messages"""
        try:
            # Check if agent_id is provided for agent-based completions
            agent_id = kwargs.get("agent_id", self.agent_id)
            
            if agent_id:
                # Use agent-based completion
                response = self.client.agents.complete(
                    messages=messages,
                    agent_id=agent_id,
                    stream=False
                )
                
                # Extract content from agent response
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    if hasattr(response.choices[0], 'message'):
                        return response.choices[0].message.content
                    elif hasattr(response.choices[0], 'content'):
                        return response.choices[0].content
                return str(response)
            else:
                # Use standard chat completion
                response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    temperature=kwargs.get("temperature", self.default_temperature),
                    max_tokens=kwargs.get("max_tokens", self.default_max_tokens),
                    stream=False
                )
                
                # Extract content from response
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    if hasattr(response.choices[0], 'message'):
                        return response.choices[0].message.content
                    elif hasattr(response.choices[0], 'content'):
                        return response.choices[0].content
                return str(response)

        except Exception as e:
            self.logger.error(f"Mistral API request failed: {e}")
            raise

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream responses from Mistral"""
        try:
            messages = [{"role": "user", "content": prompt}]
            agent_id = kwargs.get("agent_id", self.agent_id)
            
            if agent_id:
                # Use agent-based streaming
                try:
                    response = self.client.agents.complete(
                        messages=messages,
                        agent_id=agent_id,
                        stream=True
                    )
                    
                    for chunk in response:
                        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                            if hasattr(chunk.choices[0], 'delta'):
                                delta = chunk.choices[0].delta
                                if hasattr(delta, 'content') and delta.content:
                                    yield delta.content
                            elif hasattr(chunk.choices[0], 'content') and chunk.choices[0].content:
                                yield chunk.choices[0].content
                except Exception as stream_error:
                    # Fallback to non-streaming if streaming fails
                    self.logger.warning(f"Mistral streaming failed, falling back to non-streaming: {stream_error}")
                    result = self.client.agents.complete(
                        messages=messages,
                        agent_id=agent_id,
                        stream=False
                    )
                    if hasattr(result, 'choices') and len(result.choices) > 0:
                        content = ""
                        if hasattr(result.choices[0], 'message'):
                            content = result.choices[0].message.content or ""
                        elif hasattr(result.choices[0], 'content'):
                            content = result.choices[0].content or ""
                        if content:
                            yield content
            else:
                # Use standard chat streaming
                try:
                    response = self.client.chat.complete(
                        model=self.model,
                        messages=messages,
                        temperature=kwargs.get("temperature", self.default_temperature),
                        max_tokens=kwargs.get("max_tokens", self.default_max_tokens),
                        stream=True
                    )
                    
                    for chunk in response:
                        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                            if hasattr(chunk.choices[0], 'delta'):
                                delta = chunk.choices[0].delta
                                if hasattr(delta, 'content') and delta.content:
                                    yield delta.content
                            elif hasattr(chunk.choices[0], 'content') and chunk.choices[0].content:
                                yield chunk.choices[0].content
                except Exception as stream_error:
                    # Fallback to non-streaming if streaming fails
                    self.logger.warning(f"Mistral streaming failed, falling back to non-streaming: {stream_error}")
                    result = self.client.chat.complete(
                        model=self.model,
                        messages=messages,
                        temperature=kwargs.get("temperature", self.default_temperature),
                        max_tokens=kwargs.get("max_tokens", self.default_max_tokens),
                        stream=False
                    )
                    if hasattr(result, 'choices') and len(result.choices) > 0:
                        content = ""
                        if hasattr(result.choices[0], 'message'):
                            content = result.choices[0].message.content or ""
                        elif hasattr(result.choices[0], 'content'):
                            content = result.choices[0].content or ""
                        if content:
                            yield content

        except Exception as e:
            self.logger.error(f"Mistral streaming request failed: {e}")
            raise


# Register Mistral caller with the factory
LLMFactory.register_caller(LLMProvider.MISTRAL, MistralCaller)

