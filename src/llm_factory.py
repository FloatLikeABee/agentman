"""
LLM Factory Pattern for managing multiple LLM providers
Supports Gemini, Qwen, and GLM models
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Iterator
from enum import Enum
import logging


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"
    QWEN = "qwen"


class BaseLLMCaller(ABC):
    """Base class for all LLM callers"""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = kwargs
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat with the LLM using a list of messages"""
        pass
    
    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream responses from the LLM"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model": self.model,
            "provider": self.__class__.__name__,
            "config": self.config
        }


class LLMFactory:
    """Factory class for creating LLM callers"""
    
    _callers: Dict[LLMProvider, type] = {}
    _instances: Dict[str, BaseLLMCaller] = {}
    
    @classmethod
    def register_caller(cls, provider: LLMProvider, caller_class: type):
        """Register a new LLM caller"""
        if not issubclass(caller_class, BaseLLMCaller):
            raise ValueError(f"{caller_class} must inherit from BaseLLMCaller")
        cls._callers[provider] = caller_class
        logging.info(f"Registered LLM caller: {provider.value}")
    
    @classmethod
    def create_caller(
        cls,
        provider: LLMProvider,
        api_key: str,
        model: str,
        **kwargs
    ) -> BaseLLMCaller:
        """Create an LLM caller instance"""
        if provider not in cls._callers:
            raise ValueError(f"Unknown LLM provider: {provider}")
        
        # Create a unique key for caching instances
        instance_key = f"{provider.value}:{model}"
        
        # Return cached instance if exists
        if instance_key in cls._instances:
            return cls._instances[instance_key]
        
        # Create new instance
        caller_class = cls._callers[provider]
        instance = caller_class(api_key=api_key, model=model, **kwargs)
        cls._instances[instance_key] = instance
        
        logging.info(f"Created LLM caller: {provider.value} with model {model}")
        return instance
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available providers"""
        return [provider.value for provider in cls._callers.keys()]
    
    @classmethod
    def clear_cache(cls):
        """Clear cached instances"""
        cls._instances.clear()
        logging.info("Cleared LLM caller cache")

