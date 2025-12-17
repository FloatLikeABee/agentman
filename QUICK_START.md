# Quick Start Guide - LLM Factory

## Installation

No additional dependencies needed beyond what's already in `requirements.txt`.

## Running the Test Suite

```bash
python test_llm_factory.py
```

This will test all three providers (Gemini, Qwen, GLM) with basic operations.

## Starting the API Server

```bash
# From the project root
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

## Quick Examples

### 1. Check System Status

```bash
curl http://localhost:8000/status
```

Expected response:
```json
{
  "llm_providers_available": ["gemini", "qwen", "glm"],
  "default_llm_provider": "gemini",
  "available_models": [...],
  "rag_collections": [...],
  "active_agents": [],
  "active_tools": []
}
```

### 2. List Available Providers

```bash
curl http://localhost:8000/providers
```

Expected response:
```json
{
  "providers": ["gemini", "qwen", "glm"],
  "default": "gemini"
}
```

### 3. Create an Agent with Gemini

```bash
curl -X POST http://localhost:8000/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Gemini Agent",
    "description": "A helpful assistant using Gemini",
    "agent_type": "hybrid",
    "llm_provider": "gemini",
    "model_name": "gemini-2.5-flash",
    "temperature": 0.7,
    "max_tokens": 2048,
    "tools": []
  }'
```

### 4. Create an Agent with Qwen

```bash
curl -X POST http://localhost:8000/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Qwen Agent",
    "description": "A helpful assistant using Qwen",
    "agent_type": "hybrid",
    "llm_provider": "qwen",
    "model_name": "qwen3-max",
    "temperature": 0.7,
    "max_tokens": 2048,
    "tools": []
  }'
```

### 5. Create an Agent with GLM

```bash
curl -X POST http://localhost:8000/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My GLM Agent",
    "description": "A helpful assistant using GLM",
    "agent_type": "hybrid",
    "llm_provider": "glm",
    "model_name": "glm-4.6",
    "temperature": 0.7,
    "max_tokens": 2048,
    "tools": []
  }'
```

### 6. List All Agents

```bash
curl http://localhost:8000/agents
```

### 7. Run an Agent

```bash
# Replace {agent_id} with the actual agent ID from the creation response
curl -X POST http://localhost:8000/agents/{agent_id}/run \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?"
  }'
```

## Python Usage

### Direct LLM Call

```python
from src.llm_factory import LLMFactory, LLMProvider
import gemini_caller  # Import to register

# Create caller
caller = LLMFactory.create_caller(
    provider=LLMProvider.GEMINI,
    api_key="AIzaSyAt19tBj232GyyUbM95MlZzZarqZcTKmsc",
    model="gemini-2.5-flash"
)

# Generate response
response = caller.generate("What is AI?")
print(response)
```

### Using LLM Service

```python
from services.llm_service import LLMService

# Create service with Gemini
service = LLMService(provider="gemini")

# Generate answer
answer = service.generate_answer(
    context="AI is a field of computer science.",
    question="What is AI?"
)
print(answer)
```

### With LangChain Agents

```python
from src.llm_factory import LLMFactory, LLMProvider
from src.llm_langchain_wrapper import LangChainLLMWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import qwen_caller

# Create LLM
llm_caller = LLMFactory.create_caller(
    provider=LLMProvider.QWEN,
    api_key="sk-fc88e8c463e94a43bc41f1094a28fa1f",
    model="qwen3-max"
)
llm = LangChainLLMWrapper(llm_caller=llm_caller)

# Create tools
tools = [
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="Useful for math calculations"
    )
]

# Create agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run agent
result = agent.run("What is 25 * 4?")
print(result)
```

## Switching Providers

Simply change the `llm_provider` parameter when creating agents:

- `"gemini"` - Google Gemini
- `"qwen"` - Alibaba Qwen  
- `"glm"` - ChatGLM

All providers support the same interface, so switching is seamless!

## Troubleshooting

### Import Errors
Make sure you're running from the project root directory.

### API Key Errors
Check that the API keys in `src/config.py` are correct.

### Connection Errors
Verify internet connectivity and that the API endpoints are accessible.

## Next Steps

- Read `LLM_FACTORY_GUIDE.md` for detailed documentation
- Check `MIGRATION_SUMMARY.md` for technical details
- Run `test_llm_factory.py` to verify everything works

