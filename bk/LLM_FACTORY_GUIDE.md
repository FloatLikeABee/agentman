# LLM Factory Pattern Implementation Guide

## Overview

The project has been refactored from using Ollama to a flexible factory pattern that supports multiple LLM providers:
- **Gemini** (Google)
- **Qwen** (Alibaba)
- **GLM** (ChatGLM)

## Architecture

### Core Components

1. **LLM Factory** (`src/llm_factory.py`)
   - Base class `BaseLLMCaller` for all LLM implementations
   - Factory class `LLMFactory` for creating LLM instances
   - Provider enum `LLMProvider` for supported providers

2. **LLM Callers**
   - `gemini_caller.py` - Google Gemini API integration
   - `qwen_caller.py` - Alibaba Qwen API integration
   - `glm_caller.py` - ChatGLM API integration

3. **LangChain Wrapper** (`src/llm_langchain_wrapper.py`)
   - Makes custom LLM callers compatible with LangChain agents

## Configuration

### API Keys (in `src/config.py`)

```python
# Gemini
gemini_api_key: str = "AIzaSyAt19tBj232GyyUbM95MlZzZarqZcTKmsc"
gemini_default_model: str = "gemini-2.5-flash"

# Qwen
qwen_api_key: str = "sk-fc88e8c463e94a43bc41f1094a28fa1f"
qwen_default_model: str = "qwen3-max"

# GLM
glm_api_key: str = "0aa9ad7dd2114ef3bfb16bea056a977a.9nVmCTUa8l6SBCK6"
glm_default_model: str = "glm-4.6"

# Default provider
default_llm_provider: str = "gemini"
```

## Usage

### 1. Direct Usage (Without LangChain)

```python
from src.llm_factory import LLMFactory, LLMProvider
import gemini_caller  # Import to register

# Create a caller
caller = LLMFactory.create_caller(
    provider=LLMProvider.GEMINI,
    api_key="your-api-key",
    model="gemini-2.5-flash"
)

# Generate response
response = caller.generate("What is AI?")

# Chat with messages
messages = [
    {"role": "user", "content": "Hello!"}
]
response = caller.chat(messages)

# Stream responses
for chunk in caller.stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

### 2. With LangChain Agents

```python
from src.llm_factory import LLMFactory, LLMProvider
from src.llm_langchain_wrapper import LangChainLLMWrapper
from langchain.agents import initialize_agent, AgentType
import qwen_caller

# Create LLM caller
llm_caller = LLMFactory.create_caller(
    provider=LLMProvider.QWEN,
    api_key="your-api-key",
    model="qwen3-max"
)

# Wrap for LangChain
llm = LangChainLLMWrapper(llm_caller=llm_caller)

# Use with agents
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

### 3. Via REST API

#### Create an Agent with Specific Provider

```bash
curl -X POST http://localhost:8000/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Agent",
    "description": "Test agent",
    "agent_type": "hybrid",
    "llm_provider": "gemini",
    "model_name": "gemini-2.5-flash",
    "temperature": 0.7,
    "max_tokens": 2048,
    "tools": ["wikipedia"]
  }'
```

#### List Available Providers

```bash
curl http://localhost:8000/providers
```

#### List Available Models

```bash
curl http://localhost:8000/models
```

#### Check System Status

```bash
curl http://localhost:8000/status
```

## Switching Between Providers

### In Agent Configuration

When creating an agent, specify the `llm_provider` field:

```python
from src.models import AgentConfig, LLMProviderType

config = AgentConfig(
    name="Gemini Agent",
    agent_type="hybrid",
    llm_provider=LLMProviderType.GEMINI,  # or QWEN, GLM
    model_name="gemini-2.5-flash",
    temperature=0.7
)
```

### In LLM Service

```python
from services.llm_service import LLMService

# Use Gemini
service = LLMService(provider="gemini", model="gemini-2.5-flash")

# Use Qwen
service = LLMService(provider="qwen", model="qwen3-max")

# Use GLM
service = LLMService(provider="glm", model="glm-4.6")
```

## Testing

Run the test suite:

```bash
python test_llm_factory.py
```

## Migration from Ollama

All Ollama references have been replaced with the factory pattern:

1. ✅ `services/llm_service.py` - Now uses LLM factory
2. ✅ `src/agent_manager.py` - Now uses LLM factory with LangChain wrapper
3. ✅ `agent_codes.py` - Updated to use factory pattern
4. ✅ `src/api.py` - Updated endpoints to support provider selection
5. ✅ `src/config.py` - Added configuration for all providers
6. ✅ `src/models.py` - Added LLMProviderType enum

## API Endpoints Changes

- `/status` - Now returns `llm_providers_available` and `default_llm_provider`
- `/providers` - New endpoint to list available providers
- `/models` - Now returns models from all providers
- `/agents` - Now accepts `llm_provider` parameter

## Notes

- The factory pattern allows easy addition of new LLM providers
- All callers implement the same interface (`BaseLLMCaller`)
- LangChain compatibility is maintained through the wrapper
- API keys are stored in configuration (should be moved to environment variables in production)

