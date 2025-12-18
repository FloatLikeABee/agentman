from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import logging
from typing import List, Dict, Any, Optional
import asyncio

from .config import settings
from .models import (
    RAGDataInput,
    RAGDataValidation,
    AgentConfig,
    ToolConfig,
    QueryRequest,
    QueryResponse,
    SystemStatus,
    ModelInfo,
    RAGQueryRequest,
    DirectLLMRequest,
    DirectLLMResponse,
    LLMProviderType,
    CustomizationCreateRequest,
    CustomizationQueryRequest,
    CustomizationQueryResponse,
    CrawlerRequest,
    CrawlerResponse,
)
from .rag_system import RAGSystem
from .agent_manager import AgentManager
from .tools import ToolManager
from .mcp_service import MCPService
from .llm_factory import LLMFactory, LLMProvider
from .llm_langchain_wrapper import LangChainLLMWrapper
from .customization import CustomizationManager
from .crawler import CrawlerService


class RAGAPI:
    def __init__(self):
        self.app = FastAPI(
            title="RAG System API",
            description="""
            A comprehensive RAG (Retrieval-Augmented Generation) System API with MCP (Model Context Protocol) support.

            ## Features
            - **Agent Management**: Create and manage AI agents with different LLM providers
            - **RAG Collections**: Store and query knowledge bases using vector search
            - **Tool Integration**: Connect various tools and APIs to agents
            - **Direct LLM Calls**: Call LLMs directly with optional web search capability
            - **Streaming Responses**: Real-time streaming of agent responses
            - **MCP Support**: Compatible with Model Context Protocol for enhanced AI interactions

            ## Authentication
            Currently, no authentication is required for API access.

            ## Rate Limits
            No rate limiting is currently implemented.
            """,
            version="1.0.0",
            contact={
                "name": "RAG System Team",
                "email": "support@rag-system.com",
            },
            license_info={
                "name": "MIT",
            },
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
        
        # Initialize components
        self.rag_system = RAGSystem()
        self.tool_manager = ToolManager(rag_system=self.rag_system)
        self.agent_manager = AgentManager(self.rag_system, self.tool_manager)
        self.mcp_service = MCPService(self.agent_manager, self.rag_system, self.tool_manager)
        self.customization_manager = CustomizationManager()
        self.crawler_service = CrawlerService(self.rag_system)
        
        # Setup CORS - Allow all origins (configurable)
        # By default this uses settings.cors_origins which is ["*"],
        # meaning any frontend origin (localhost, 127.0.0.1, any port) is allowed.
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_origin_regex=settings.cors_origin_regex,
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {"message": "RAG System API", "version": "1.0.0"}

        @self.app.get("/status", tags=["System"])
        async def get_status() -> SystemStatus:
            """Get comprehensive system status including available providers, models, collections, agents, and tools."""
            try:
                llm_providers = self.agent_manager.get_available_providers()
                available_models = self.agent_manager.get_available_models()
                rag_collections = [col['name'] for col in self.rag_system.list_collections()]
                active_agents = [agent['id'] for agent in self.agent_manager.list_agents()]
                active_tools = [tool['id'] for tool in self.tool_manager.list_tools()]

                return SystemStatus(
                    llm_providers_available=llm_providers,
                    default_llm_provider=settings.default_llm_provider,
                    available_models=available_models,
                    rag_collections=rag_collections,
                    active_agents=active_agents,
                    active_tools=active_tools
                )
            except Exception as e:
                self.logger.error(f"Error getting status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # RAG Endpoints
        @self.app.post("/rag/validate", tags=["RAG"])
        async def validate_rag_data(data_input: RAGDataInput) -> RAGDataValidation:
            """Validate RAG data input before adding to collections."""
            try:
                return self.rag_system.validate_data(data_input)
            except Exception as e:
                self.logger.error(f"Error validating data: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/rag/collections/{collection_name}/data", tags=["RAG"])
        async def add_rag_data(collection_name: str, data_input: RAGDataInput):
            """Add data to a specific RAG collection for vector search and retrieval."""
            try:
                success = self.rag_system.add_data_to_collection(collection_name, data_input)
                if success:
                    return {"message": f"Data added to collection {collection_name}"}
                else:
                    raise HTTPException(status_code=400, detail="Failed to add data")
            except Exception as e:
                self.logger.error(f"Error adding data: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/rag/collections", tags=["RAG"])
        async def list_rag_collections():
            """List all available RAG collections with their metadata and document counts."""
            try:
                return self.rag_system.list_collections()
            except Exception as e:
                self.logger.error(f"Error listing collections: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/rag/collections/{collection_name}/query", tags=["RAG"])
        async def query_rag_collection(collection_name: str, request: RAGQueryRequest):
            """Query a specific RAG collection using semantic search to find relevant documents."""
            try:
                results = self.rag_system.query_collection(collection_name, request.query, request.n_results)
                return {"results": results}
            except Exception as e:
                self.logger.error(f"Error querying collection: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/rag/collections/{collection_name}", tags=["RAG"])
        async def delete_rag_collection(collection_name: str):
            """Delete a RAG collection and all its associated data."""
            try:
                success = self.rag_system.delete_collection(collection_name)
                if success:
                    return {"message": f"Collection {collection_name} deleted"}
                else:
                    raise HTTPException(status_code=400, detail="Failed to delete collection")
            except Exception as e:
                self.logger.error(f"Error deleting collection: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Agent Endpoints
        @self.app.post("/agents", tags=["Agents"])
        async def create_agent(config: AgentConfig):
            """Create a new AI agent with specified configuration including LLM provider, model, tools, and RAG collections."""
            try:
                agent_id = self.agent_manager.create_agent(config)
                return {"agent_id": agent_id, "message": "Agent created successfully"}
            except Exception as e:
                self.logger.error(f"Error creating agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/agents", tags=["Agents"])
        async def list_agents():
            """List all configured agents with their current status and configurations."""
            try:
                return self.agent_manager.list_agents()
            except Exception as e:
                self.logger.error(f"Error listing agents: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/agents/{agent_id}", tags=["Agents"])
        async def get_agent(agent_id: str):
            """Get detailed information about a specific agent including its configuration and status."""
            try:
                agent_data = self.agent_manager.get_agent(agent_id)
                if agent_data:
                    # Get config and convert to dict if it's a Pydantic model
                    config = agent_data.get('config')
                    config_dict = {}
                    
                    try:
                        if config and hasattr(config, 'model_dump'):
                            # Use model_dump with mode='python' to get plain Python types
                            config_dict = config.model_dump(mode='python')
                        elif config and hasattr(config, 'dict'):
                            config_dict = config.dict()
                        elif isinstance(config, dict):
                            config_dict = config.copy()
                        elif config:
                            # Fallback: manually extract fields
                            config_dict = {
                                'name': getattr(config, 'name', 'Unknown') if hasattr(config, 'name') else 'Unknown',
                                'description': getattr(config, 'description', '') if hasattr(config, 'description') else '',
                                'agent_type': str(getattr(config, 'agent_type', 'rag')) if hasattr(config, 'agent_type') else 'rag',
                                'llm_provider': str(getattr(config, 'llm_provider', 'gemini')) if hasattr(config, 'llm_provider') else 'gemini',
                                'model_name': getattr(config, 'model_name', '') if hasattr(config, 'model_name') else '',
                                'temperature': float(getattr(config, 'temperature', 0.7)) if hasattr(config, 'temperature') else 0.7,
                                'max_tokens': int(getattr(config, 'max_tokens', 8192)) if hasattr(config, 'max_tokens') else 8192,
                                'rag_collections': list(getattr(config, 'rag_collections', [])) if hasattr(config, 'rag_collections') else [],
                                'tools': list(getattr(config, 'tools', [])) if hasattr(config, 'tools') else [],
                                'system_prompt': getattr(config, 'system_prompt', '') if hasattr(config, 'system_prompt') else '',
                                'is_active': bool(getattr(config, 'is_active', True)) if hasattr(config, 'is_active') else True,
                            }
                    except Exception as config_error:
                        self.logger.warning(f"Error converting config to dict: {config_error}, using fallback")
                        config_dict = {
                            'name': 'Unknown',
                            'description': '',
                            'agent_type': 'rag',
                            'llm_provider': 'gemini',
                            'model_name': '',
                            'temperature': 0.7,
                            'max_tokens': 8192,
                            'rag_collections': [],
                            'tools': [],
                            'system_prompt': '',
                            'is_active': True,
                        }
                    
                    # Ensure all values are JSON-serializable (convert enums, etc.)
                    if 'llm_provider' in config_dict and hasattr(config_dict['llm_provider'], 'value'):
                        config_dict['llm_provider'] = config_dict['llm_provider'].value
                    if 'agent_type' in config_dict and hasattr(config_dict['agent_type'], 'value'):
                        config_dict['agent_type'] = config_dict['agent_type'].value
                    
                    # Return only serializable data, exclude runtime objects
                    return {
                        'id': agent_id,
                        'config': config_dict,
                        'provider': str(agent_data.get('provider', 'Unknown')),
                        'model': str(agent_data.get('model', 'Unknown')),
                        # Exclude 'agent', 'llm', 'llm_caller', and 'rag_system' as they are not serializable
                    }
                else:
                    raise HTTPException(status_code=404, detail="Agent not found")
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting agent: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Error retrieving agent: {str(e)}")

        @self.app.put("/agents/{agent_id}", tags=["Agents"])
        async def update_agent(agent_id: str, config: AgentConfig):
            """Update an existing agent's configuration. The agent will be recreated with the new settings."""
            try:
                success = self.agent_manager.update_agent(agent_id, config)
                if success:
                    return {"message": "Agent updated successfully"}
                else:
                    raise HTTPException(status_code=400, detail="Failed to update agent")
            except Exception as e:
                self.logger.error(f"Error updating agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/agents/{agent_id}", tags=["Agents"])
        async def delete_agent(agent_id: str):
            """Delete an agent and remove it from the system permanently."""
            try:
                success = self.agent_manager.delete_agent(agent_id)
                if success:
                    return {"message": "Agent deleted successfully"}
                else:
                    raise HTTPException(status_code=400, detail="Failed to delete agent")
            except Exception as e:
                self.logger.error(f"Error deleting agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/agents/{agent_id}/run", tags=["Agents"])
        async def run_agent(agent_id: str, request: QueryRequest):
            """Execute an agent with a query and optional context. Returns the agent's response."""
            try:
                result = await self.agent_manager.run_agent(
                    agent_id,
                    request.query,
                    request.context
                )
                return QueryResponse(
                    response=result['response'],
                    sources=result.get('sources', []),
                    metadata=result.get('metadata', {})
                )
            except Exception as e:
                self.logger.error(f"Error running agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/agents/{agent_id}/run/stream", tags=["Agents"])
        async def run_agent_stream(agent_id: str, request: QueryRequest):
            """Execute an agent with streaming response. Returns a stream of text chunks as the agent generates its response."""
            try:
                return StreamingResponse(
                    self.agent_manager.run_agent_stream(
                        agent_id,
                        request.query,
                        request.context
                    ),
                    media_type="text/plain"
                )
            except Exception as e:
                self.logger.error(f"Error running streaming agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Direct LLM Endpoints
        @self.app.post("/llm/direct", tags=["Direct LLM"])
        async def call_llm_direct(request: DirectLLMRequest) -> DirectLLMResponse:
            """Call LLM directly with optional web search capability. Allows specifying model and parameters directly."""
            try:
                # Determine provider from model name
                if request.model_name.startswith('gemini'):
                    provider = LLMProviderType.GEMINI
                    api_key = settings.gemini_api_key
                elif request.model_name.startswith('qwen'):
                    provider = LLMProviderType.QWEN
                    api_key = settings.qwen_api_key
                else:
                    # Default to Gemini if model not recognized
                    provider = LLMProviderType.GEMINI
                    api_key = settings.gemini_api_key
                    self.logger.warning(f"Unrecognized model {request.model_name}, defaulting to Gemini")

                # Create LLM caller
                llm_caller = LLMFactory.create_caller(
                    provider=LLMProvider(provider.value),
                    api_key=api_key,
                    model=request.model_name,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )

                # Wrap in LangChain-compatible wrapper
                llm = LangChainLLMWrapper(llm_caller=llm_caller)

                # Prepare tools
                tools = []
                web_search_used = False

                if request.use_web_search:
                    # Add web search tool
                    search_tool = self.tool_manager.get_tool("web_search")
                    if search_tool:
                        tools.append(search_tool)
                        web_search_used = True

                # Create system prompt
                system_prompt = request.system_prompt or "You are a helpful AI assistant."

                if tools:
                    # Use agent with tools
                    from langchain.agents import AgentExecutor, create_react_agent
                    from langchain.prompts import PromptTemplate

                    # Create ReAct prompt template
                    react_template = f"""{system_prompt}

You have access to the following tools:

{{tools}}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
{{agent_scratchpad}}"""

                    prompt = PromptTemplate(
                        input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
                        template=react_template
                    )

                    # Create the agent
                    agent_prompt = create_react_agent(llm, tools, prompt)

                    # Wrap in AgentExecutor
                    agent = AgentExecutor(
                        agent=agent_prompt,
                        tools=tools,
                        verbose=True,
                        max_iterations=10,
                        early_stopping_method='force',
                        return_intermediate_steps=False,
                        handle_parsing_errors=True,
                    )

                    # Run agent using async invoke to use the fixed async LLM path
                    try:
                        # Use ainvoke instead of invoke to use the async path that properly handles Generation objects
                        response = await agent.ainvoke({"input": request.query})
                    except AttributeError as e:
                        print(e)
                        if "'str' object has no attribute 'text'" in str(e):
                            # This error occurs when the agent's LLM returns strings instead of Generation objects
                            # Try using the LLM directly as a fallback
                            self.logger.warning(
                                f"Agent executor failed due to LLM wrapper issue: {e}. "
                                "Falling back to direct LLM call."
                            )
                            # Fall back to direct LLM call without tools
                            response_text = await llm.apredict(f"{system_prompt}\n\nQuery: {request.query}")
                            web_search_used = False  # Reset since we're not using tools
                        else:
                            raise
                    except Exception as e:
                        # Catch any other errors and provide better error message
                        self.logger.error(f"Agent executor error: {e}")
                        raise
                    else:
                        # Extract response
                        if isinstance(response, dict) and 'output' in response:
                            response_text = response['output']
                        elif isinstance(response, str):
                            response_text = response
                        else:
                            response_text = str(response)
                else:
                    # Direct LLM call without tools
                    # Add timeout to prevent hanging requests
                    import asyncio
                    try:
                        response_text = await asyncio.wait_for(
                            llm.apredict(f"{system_prompt}\n\nQuery: {request.query}"),
                            timeout=settings.api_timeout
                        )
                    except asyncio.TimeoutError:
                        raise TimeoutError(
                            f"LLM API request timed out after {settings.api_timeout} seconds. "
                            f"The API may be slow or unavailable. Please try again."
                        )

                return DirectLLMResponse(
                    response=response_text,
                    model_used=request.model_name,
                    web_search_used=web_search_used,
                    metadata={
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens,
                        "provider": provider.value
                    }
                )

            except ConnectionError as e:
                self.logger.error(f"Network error calling LLM directly: {e}")
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Service Unavailable",
                        "message": str(e),
                        "suggestion": "Please check your network connection and try again. If using Gemini API, ensure you can reach generativelanguage.googleapis.com"
                    }
                )
            except TimeoutError as e:
                self.logger.error(f"Timeout calling LLM directly: {e}")
                raise HTTPException(
                    status_code=504,
                    detail={
                        "error": "Gateway Timeout",
                        "message": str(e),
                        "suggestion": "The LLM API request took too long. Please try again with a simpler query or check if the API service is available."
                    }
                )
            except Exception as e:
                self.logger.error(f"Error calling LLM directly: {e}")
                error_detail = str(e)
                # Provide more context for common errors
                if "DNS" in error_detail or "503" in error_detail:
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "error": "Service Unavailable",
                            "message": error_detail,
                            "suggestion": "Unable to reach the LLM API. Please check your network connection and API configuration."
                        }
                    )
                raise HTTPException(status_code=500, detail=str(e))

        # Tool Endpoints
        @self.app.get("/tools", tags=["Tools"])
        async def list_tools():
            """List all available tools that can be used by agents, including their configurations and status."""
            try:
                return self.tool_manager.list_tools()
            except Exception as e:
                self.logger.error(f"Error listing tools: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put("/tools/{tool_id}", tags=["Tools"])
        async def update_tool_config(tool_id: str, config: ToolConfig):
            """Update the configuration of a specific tool. Changes take effect immediately."""
            try:
                success = self.tool_manager.update_tool_config(tool_id, config)
                if success:
                    return {"message": "Tool configuration updated successfully"}
                else:
                    raise HTTPException(status_code=400, detail="Failed to update tool configuration")
            except Exception as e:
                self.logger.error(f"Error updating tool config: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Model Endpoints
        @self.app.get("/models", tags=["Models"])
        async def list_models():
            """List all available LLM models from configured providers with their specifications."""
            try:
                return self.agent_manager.get_available_models()
            except Exception as e:
                self.logger.error(f"Error listing models: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/providers", tags=["Models"])
        async def list_providers():
            """List all configured LLM providers and their current status."""
            try:
                return {
                    "providers": self.agent_manager.get_available_providers(),
                    "default": settings.default_llm_provider
                }
            except Exception as e:
                self.logger.error(f"Error listing providers: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # MCP Endpoints
        @self.app.post("/mcp/start", tags=["MCP"])
        async def start_mcp_server(background_tasks: BackgroundTasks):
            """Start the Model Context Protocol (MCP) server for enhanced AI interactions."""
            try:
                background_tasks.add_task(self.mcp_service.start_server)
                return {"message": "MCP server starting"}
            except Exception as e:
                self.logger.error(f"Error starting MCP server: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Customization Endpoints
        @self.app.post("/customizations", tags=["Customizations"])
        async def create_customization(req: CustomizationCreateRequest):
            """Create a new customization profile (instructions + optional RAG/LLM config)."""
            try:
                profile_id = self.customization_manager.create_profile(req)
                return {"id": profile_id, "message": "Customization created successfully"}
            except Exception as e:
                self.logger.error(f"Error creating customization: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/customizations", tags=["Customizations"])
        async def list_customizations():
            """List all customization profiles."""
            try:
                return [p.model_dump() for p in self.customization_manager.list_profiles()]
            except Exception as e:
                self.logger.error(f"Error listing customizations: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/customizations/{profile_id}", tags=["Customizations"])
        async def get_customization(profile_id: str):
            """Get a single customization profile by id."""
            try:
                profile = self.customization_manager.get_profile(profile_id)
                if not profile:
                    raise HTTPException(status_code=404, detail="Customization not found")
                return profile
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting customization {profile_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put("/customizations/{profile_id}", tags=["Customizations"])
        async def update_customization(profile_id: str, req: CustomizationCreateRequest):
            """Update an existing customization profile."""
            try:
                success = self.customization_manager.update_profile(profile_id, req)
                if success:
                    return {"message": "Customization updated successfully"}
                raise HTTPException(status_code=404, detail="Customization not found")
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error updating customization {profile_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/customizations/{profile_id}", tags=["Customizations"])
        async def delete_customization(profile_id: str):
            """Delete a customization profile."""
            try:
                success = self.customization_manager.delete_profile(profile_id)
                if success:
                    return {"message": "Customization deleted successfully"}
                raise HTTPException(status_code=404, detail="Customization not found")
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error deleting customization {profile_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/customizations/{profile_id}/query", tags=["Customizations"])
        async def query_customization(profile_id: str, request: CustomizationQueryRequest) -> CustomizationQueryResponse:
            """
            Run a short user query through a customization profile.

            Combines the profile's system_prompt with optional RAG context, then calls the LLM.
            """
            try:
                profile = self.customization_manager.get_profile(profile_id)
                if not profile:
                    raise HTTPException(status_code=404, detail="Customization not found")

                # Determine provider/model
                provider_str = (
                    profile.llm_provider.value
                    if profile.llm_provider
                    else settings.default_llm_provider
                )
                if provider_str == "gemini":
                    provider = LLMProviderType.GEMINI
                    api_key = settings.gemini_api_key
                    model_name = profile.model_name or settings.gemini_default_model
                elif provider_str == "qwen":
                    provider = LLMProviderType.QWEN
                    api_key = settings.qwen_api_key
                    model_name = profile.model_name or settings.qwen_default_model
                else:
                    # Fallback to default provider
                    provider = LLMProviderType.GEMINI
                    api_key = settings.gemini_api_key
                    model_name = profile.model_name or settings.gemini_default_model

                # Create LLM caller
                llm_caller = LLMFactory.create_caller(
                    provider=LLMProvider(provider.value),
                    api_key=api_key,
                    model=model_name,
                    temperature=request.temperature if request.temperature is not None else 0.7,
                    max_tokens=request.max_tokens if request.max_tokens is not None else 8192,
                )

                # Wrap in LangChain-compatible wrapper
                llm = LangChainLLMWrapper(llm_caller=llm_caller)

                # Build context from RAG collection if specified
                context = ""
                rag_used: Optional[str] = None
                if profile.rag_collection:
                    rag_used = profile.rag_collection
                    results = self.rag_system.query_collection(
                        profile.rag_collection,
                        request.query,
                        request.n_results,
                    )
                    if results:
                        context = "\n\n".join(r["content"] for r in results[: request.n_results])

                # Build final prompt
                system_prompt = profile.system_prompt
                if context:
                    full_prompt = (
                        f"{system_prompt}\n\n"
                        f"Context (from knowledge base '{rag_used}'):\n{context}\n\n"
                        f"User query:\n{request.query}"
                    )
                else:
                    full_prompt = f"{system_prompt}\n\nUser query:\n{request.query}"

                # Direct LLM call (no tools for now)
                response_text = await llm.apredict(full_prompt)

                return CustomizationQueryResponse(
                    response=response_text,
                    profile_id=profile.id,
                    profile_name=profile.name,
                    model_used=model_name,
                    rag_collection_used=rag_used,
                    metadata={
                        "temperature": request.temperature if request.temperature is not None else 0.7,
                        "max_tokens": request.max_tokens if request.max_tokens is not None else 8192,
                        "provider": provider.value,
                    },
                )
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error querying customization {profile_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Crawler Endpoints
        @self.app.post("/crawler/crawl", tags=["Crawler"], response_model=CrawlerResponse)
        async def crawl_website(request: CrawlerRequest) -> CrawlerResponse:
            """
            Crawl a website, extract content with AI, and save to RAG collection.
            AI will automatically generate collection name and description if not provided.
            """
            try:
                result = self.crawler_service.crawl_and_save(
                    url=request.url,
                    use_js=request.use_js,
                    llm_provider=request.llm_provider,
                    model=request.model,
                    collection_name=request.collection_name,
                    collection_description=request.collection_description
                )
                
                if result.get("success"):
                    return CrawlerResponse(
                        success=True,
                        url=result["url"],
                        collection_name=result.get("collection_name"),
                        collection_description=result.get("collection_description"),
                        raw_file=result.get("raw_file"),
                        extracted_file=result.get("extracted_file"),
                        extracted_data=result.get("extracted_data")
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=result.get("error", "Crawling failed")
                    )
            except Exception as e:
                self.logger.error(f"Error crawling website: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=str(e))

    def get_app(self) -> FastAPI:
        """Get the FastAPI application"""
        return self.app


# Create API instance
api = RAGAPI()
app = api.get_app() 