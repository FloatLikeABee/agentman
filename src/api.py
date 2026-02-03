from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.exceptions import RequestValidationError
from fastapi import status
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import sys
from typing import List, Dict, Any, Optional
import asyncio
import os

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
    GatheringRequest,
    GatheringResponse,
    LLMProviderType,
    CustomizationCreateRequest,
    CustomizationQueryRequest,
    CustomizationQueryResponse,
    CrawlerRequest,
    CrawlerResponse,
    CrawlerProfile,
    CrawlerCreateRequest,
    CrawlerUpdateRequest,
    DatabaseToolProfile,
    DatabaseToolCreateRequest,
    DatabaseToolUpdateRequest,
    DatabaseToolPreviewResponse,
    DatabaseToolExecuteRequest,
    DatabaseType,
    RequestProfile,
    RequestCreateRequest,
    RequestUpdateRequest,
    RequestExecuteResponse,
    RequestType,
    HTTPMethod,
    DialogueProfile,
    DialogueCreateRequest,
    DialogueUpdateRequest,
    DialogueStartRequest,
    DialogueContinueRequest,
    DialogueResponse,
    DialogueMessage,
    ConversationConfig,
    ConversationCreateRequest,
    ConversationStartRequest,
    ConversationMessage,
    ConversationResponse,
    ConversationHistoryResponse,
    ConversationTurnRequest,
    SpecialFlow1Profile,
    SpecialFlow1CreateRequest,
    SpecialFlow1UpdateRequest,
    SpecialFlow1ExecuteRequest,
    SpecialFlow1ExecuteResponse,
    SmartImportRequest,
    SmartImportResponse,
)
from .rag_system import RAGSystem
from .agent_manager import AgentManager
from .tools import ToolManager
from .mcp_service import MCPService
from .llm_factory import LLMFactory, LLMProvider
from .llm_langchain_wrapper import LangChainLLMWrapper
from .customization import CustomizationManager
from .dialogue import DialogueManager
from .conversation import ConversationManager
from .crawler import CrawlerService
from .crawler_manager import CrawlerManager
from .gathering_service import GatheringService
from .db_tools import DatabaseToolsManager
from .request_tools import RequestToolsManager
from .image_reader import ImageReader
from .pdf_reader import PDFReader


class RAGAPI:
    def __init__(self):
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Define lifespan handler for graceful startup and shutdown
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("Starting Ground Control API...")
            try:
                yield
            finally:
                # Shutdown - handle graceful shutdown
                logger.info("Shutting down Ground Control API...")
                try:
                    # Get all tasks except the current one
                    current_task = asyncio.current_task()
                    tasks = [task for task in asyncio.all_tasks() if task is not current_task]
                    
                    if tasks:
                        # Cancel all tasks
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        
                        # Wait for tasks to complete cancellation
                        # Use return_exceptions=True to prevent CancelledError from propagating
                        # Set a timeout to avoid hanging
                        try:
                            results = await asyncio.wait_for(
                                asyncio.gather(*tasks, return_exceptions=True),
                                timeout=1.0  # Reduced timeout for faster shutdown
                            )
                            # Log any non-CancelledError exceptions
                            for i, result in enumerate(results):
                                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                                    logger.warning(f"Task {i} raised exception during shutdown: {result}")
                        except asyncio.TimeoutError:
                            # Timeout is expected during fast shutdown - don't log as warning
                            logger.debug("Shutdown timeout (expected during fast shutdown)")
                        except asyncio.CancelledError:
                            # This is expected if the shutdown itself is cancelled
                            logger.debug("Shutdown cancelled (expected during fast shutdown)")
                    
                    logger.info("Shutdown complete")
                except asyncio.CancelledError:
                    # CancelledError is expected during shutdown - suppress traceback
                    logger.debug("Shutdown process cancelled (expected during fast shutdown)")
                except Exception as e:
                    # Log but don't raise - shutdown should be graceful
                    # Only log non-CancelledError exceptions
                    if not isinstance(e, asyncio.CancelledError):
                        logger.warning(f"Error during shutdown: {e}")
                    else:
                        logger.debug("Shutdown cancelled (expected)")
        
        self.app = FastAPI(
            title="Ground Control API",
            lifespan=lifespan,
            description="""
            # Ground Control API
            
            A comprehensive, production-ready RAG (Retrieval-Augmented Generation) System API with advanced AI agent orchestration, knowledge management, and Model Context Protocol (MCP) support.
            
            ## 🚀 Core Capabilities
            
            ### Agent Management
            Create, configure, and manage intelligent AI agents with:
            - Multiple LLM provider support (Gemini, Qwen)
            - Customizable behavior via system prompts
            - Tool integration for extended capabilities
            - RAG integration for knowledge-augmented responses
            - Streaming and non-streaming execution modes
            
            ### RAG (Retrieval-Augmented Generation)
            Build and query knowledge bases with:
            - Vector-based semantic search
            - Multiple data format support (JSON, CSV, TXT, PDF, DOCX)
            - Automatic content chunking and embedding
            - High-performance similarity search
            - Metadata and tagging support
            
            ### Tool Ecosystem
            Extensible tool system including:
            - **Web Search**: Real-time internet information retrieval
            - **Wikipedia**: Factual information from Wikipedia
            - **Calculator**: Mathematical computations
            - **Email**: SMTP-based email sending
            - **Financial**: Real-time stock prices and financial data (yfinance/Yahoo Finance - most reliable, free, no API key)
            - **Crawler Service**: Standalone website content extraction with AI organization (available via /crawler endpoint)
            - **Decision Equalizer**: AI-powered decision-making assistance
            - **Custom Tools**: User-defined tool integration
            
            ### Direct LLM Access
            Direct language model access with:
            - Model selection and parameter control
            - Optional web search integration
            - Custom system prompts
            - Temperature and token limit configuration
            
            ### Customization Profiles
            Reusable AI behavior templates with:
            - Custom system prompts
            - RAG collection associations
            - Model and provider overrides
            - Template-based query execution
            
            ### Model Context Protocol (MCP)
            WebSocket-based protocol for:
            - Real-time AI interactions
            - Enhanced context management
            - Bidirectional communication
            - Tool execution via protocol
            
            ## 📚 API Organization
            
            The API is organized into logical groups:
            
            - **System**: Health checks and system status
            - **RAG**: Knowledge base management and querying
            - **Agents**: AI agent lifecycle and execution
            - **Direct LLM**: Direct model access
            - **Tools**: Tool management and configuration
            - **Models**: Model and provider information
            - **MCP**: Model Context Protocol server
            - **Customizations**: Behavior template management
            - **Crawler**: Website crawling and content extraction
            - **Conversations**: Multi-AI conversation (two models conversing)
            - **Image Reader**: OCR from images (Qwen Vision) and AI processing
            - **PDF Reader**: PDF text extraction and AI processing
            - **Gathering**: AI-powered data gathering from Wikipedia, Reddit, and web search
            
            ## 🔧 Technical Details
            
            ### LLM Providers
            - **Google Gemini**: High-performance multimodal models
            - **Alibaba Qwen**: Advanced language understanding models
            
            ### Vector Database
            - ChromaDB for persistent vector storage
            - Sentence transformers for embeddings
            - Configurable chunking strategies
            
            ### Agent Framework
            - LangChain-based agent orchestration
            - ReAct (Reasoning + Acting) pattern
            - Tool calling and RAG integration
            - Streaming response support
            
            ## 🔐 Security & Configuration
            
            ### Authentication
            Currently, no authentication is required. For production deployments, implement authentication middleware.
            
            ### Rate Limiting
            No rate limiting is currently implemented. Consider implementing rate limiting for production use.
            
            ### CORS
            Configurable CORS settings support cross-origin requests. Default allows all origins.
            
            ## 📖 Getting Started
            
            1. **Check System Status**: `GET /status` - Verify system health and available resources
            2. **Create RAG Collection**: `POST /rag/collections/{name}/data` - Add knowledge base content
            3. **Create Agent**: `POST /agents` - Configure an AI agent with tools and RAG
            4. **Execute Agent**: `POST /agents/{id}/run` - Run queries through your agent
            5. **Query RAG**: `POST /rag/collections/{name}/query` - Direct knowledge base search
            
            ## 🎯 Use Cases
            
            - **Customer Support**: AI agents with FAQ knowledge bases
            - **Research Assistants**: Document analysis and information retrieval
            - **Content Management**: Automated content organization and search
            - **Decision Support**: AI-powered decision-making tools
            - **Knowledge Bases**: Enterprise knowledge management systems
            - **Documentation Systems**: Searchable technical documentation
            
            ## 📝 API Documentation
            
            This API provides comprehensive Swagger/OpenAPI documentation. Access interactive documentation at:
            - **Swagger UI**: `/docs` - Interactive API explorer
            - **ReDoc**: `/redoc` - Alternative documentation interface
            - **OpenAPI Schema**: `/openapi.json` - Machine-readable API specification
            
            All endpoints include detailed descriptions, request/response examples, and parameter documentation.
            """,
            version="1.0.0",
            contact={
                "name": "Ground Control Team",
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
        self.crawler_manager = CrawlerManager()
        self.gathering_service = GatheringService(tool_manager=self.tool_manager)
        self.db_tools_manager = DatabaseToolsManager()
        self.request_tools_manager = RequestToolsManager(api_instance=self)
        self.dialogue_manager = DialogueManager(
            rag_system=self.rag_system,
            db_tools_manager=self.db_tools_manager,
            request_tools_manager=self.request_tools_manager
        )
        self.conversation_manager = ConversationManager(
            rag_system=self.rag_system,
            db_tools_manager=self.db_tools_manager,
            request_tools_manager=self.request_tools_manager
        )
        
        # Initialize Flow Service
        from .flow import FlowService
        self.flow_service = FlowService(
            customization_manager=self.customization_manager,
            agent_manager=self.agent_manager,
            db_tools_manager=self.db_tools_manager,
            request_tools_manager=self.request_tools_manager,
            crawler_service=self.crawler_service,
            rag_system=self.rag_system,
            dialogue_manager=self.dialogue_manager,
        )
        
        # Initialize Dialogue-Driven Flow Service
        from .flow import SpecialFlow1Service
        self.special_flow_1_service = SpecialFlow1Service(
            db_tools_manager=self.db_tools_manager,
            request_tools_manager=self.request_tools_manager,
            dialogue_manager=self.dialogue_manager,
            rag_system=self.rag_system,
        )
        
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
        
        # Logger already set up above
        self.logger = logger
        
        # Add exception handler for validation errors to log details
        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            """Handle Pydantic validation errors with detailed logging"""
            body = await request.body()
            self.logger.error(f"Validation error on {request.method} {request.url.path}")
            self.logger.error(f"Validation errors: {exc.errors()}")
            try:
                body_str = body.decode('utf-8') if body else "No body"
                self.logger.error(f"Request body: {body_str}")
            except:
                self.logger.error(f"Request body (raw): {body}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"detail": exc.errors()},
            )
        
        # Setup routes
        self._setup_routes()
        
        # Setup image generation routes
        self._setup_image_generation_routes()
        
        # Setup browser automation routes
        self._setup_browser_automation_routes()
        
        # Setup image reader routes
        self._setup_image_reader_routes()
        
        # Setup PDF reader routes
        self._setup_pdf_reader_routes()
        
        # Setup gathering routes
        self._setup_gathering_routes()
        
        # Setup static file handlers for common browser requests
        self._setup_static_handlers()

    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get(
            "/",
            tags=["System"],
            summary="API Root",
            description="Root endpoint that returns basic API information and version.",
            response_description="API information including name and version number."
        )
        async def root():
            """
            **API Root Endpoint**
            
            Returns basic information about the Ground Control API including the current version.
            This endpoint can be used for health checks and API discovery.
            """
            return {"message": "Ground Control API", "version": "1.0.0"}

        @self.app.get(
            "/status",
            tags=["System"],
            summary="Get System Status",
            description="Retrieve comprehensive system status including available LLM providers, models, RAG collections, active agents, and configured tools.",
            response_model=SystemStatus,
            response_description="Complete system status with all available resources and configurations."
        )
        async def get_status() -> SystemStatus:
            """
            **Get Comprehensive System Status**
            
            Returns a detailed overview of the entire system state including:
            
            - **LLM Providers**: List of available language model providers (e.g., Gemini, Qwen)
            - **Available Models**: All configured models with their specifications
            - **RAG Collections**: Knowledge bases available for retrieval
            - **Active Agents**: Currently configured and active AI agents
            - **Available Tools**: Tools that can be used by agents
            
            This endpoint is useful for:
            - System monitoring and health checks
            - Discovering available resources
            - Debugging configuration issues
            - Understanding system capabilities
            
            **Example Use Cases:**
            - Check if required providers are configured
            - Verify RAG collections are loaded
            - Monitor active agent count
            """
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
        @self.app.post(
            "/rag/validate",
            tags=["RAG"],
            summary="Validate RAG Data",
            description="Validate RAG data input format, structure, and content before adding to collections. Ensures data quality and compatibility.",
            response_model=RAGDataValidation,
            response_description="Validation results including validity status, errors, warnings, and record count."
        )
        async def validate_rag_data(data_input: RAGDataInput) -> RAGDataValidation:
            """
            **Validate RAG Data Input**
            
            Performs comprehensive validation on RAG data before it's added to a collection. This includes:
            
            - **Format Validation**: Verifies the data format (JSON, CSV, TXT, PDF, DOCX) is correct
            - **Structure Validation**: Checks data structure matches expected schema
            - **Content Validation**: Ensures content is parseable and meaningful
            - **Record Counting**: Counts records/documents in the data
            
            **Validation Checks:**
            - JSON structure and syntax
            - CSV column consistency
            - Text content quality
            - File format compatibility
            
            **Returns:**
            - `is_valid`: Boolean indicating if data passes all validation checks
            - `errors`: List of critical errors that prevent data ingestion
            - `warnings`: List of warnings that don't prevent ingestion but should be reviewed
            - `record_count`: Number of records/documents found in the data
            
            **Best Practice:** Always validate data before adding to collections to catch issues early.
            """
            try:
                return self.rag_system.validate_data(data_input)
            except Exception as e:
                self.logger.error(f"Error validating data: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/rag/suggest-title",
            tags=["RAG"],
            summary="Suggest Topic Title",
            description="Use AI to generate a short topic title from content (e.g. for RAG document name).",
            response_description="Suggested title string.",
            responses={
                200: {"description": "Suggested title", "content": {"application/json": {"example": {"title": "Key Points from Document"}}}},
                400: {"description": "Content is required or empty"},
                500: {"description": "LLM error"}
            }
        )
        async def suggest_rag_title(body: dict = Body(...)):
            """Generate a short topic title from content using the default LLM (Qwen preferred)."""
            try:
                content = (body.get("content") or "").strip()
                if not content:
                    raise HTTPException(status_code=400, detail="Content is required")
                from .llm_factory import LLMFactory, LLMProvider
                # Prefer Qwen for suggest-title when configured; else use settings default
                provider_str = "qwen" if (getattr(settings, "qwen_api_key", None) or "").strip() else (settings.default_llm_provider or "qwen")
                provider_str = provider_str.lower()
                if provider_str == "qwen":
                    provider = LLMProvider.QWEN
                    api_key = settings.qwen_api_key
                    model = settings.qwen_default_model
                elif provider_str == "gemini":
                    provider = LLMProvider.GEMINI
                    api_key = settings.gemini_api_key
                    model = settings.gemini_default_model
                elif provider_str == "mistral":
                    provider = LLMProvider.MISTRAL
                    api_key = settings.mistral_api_key
                    model = settings.mistral_default_model
                else:
                    provider = LLMProvider.QWEN
                    api_key = settings.qwen_api_key
                    model = settings.qwen_default_model
                if not api_key:
                    raise HTTPException(status_code=503, detail=f"{provider_str.capitalize()} API key not configured")
                caller = LLMFactory.create_caller(provider=provider, api_key=api_key, model=model, temperature=0.3, max_tokens=128)
                excerpt = content[:4000] if len(content) > 4000 else content
                prompt = f"""Based on the following content, suggest a short topic title (3 to 10 words). Reply with ONLY the title, no quotes or explanation.

Content:
{excerpt}"""
                title = (caller.generate(prompt) or "").strip().strip('"\'')
                if not title:
                    title = "Untitled"
                return {"title": title[:200]}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error suggesting title: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/rag/collections/{collection_name}/data",
            tags=["RAG"],
            summary="Add Data to RAG Collection",
            description="Add structured data to a RAG collection. The data will be processed, chunked, embedded, and indexed for semantic search.",
            response_description="Confirmation message indicating successful data addition.",
            responses={
                200: {
                    "description": "Data successfully added to collection",
                    "content": {
                        "application/json": {
                            "example": {"message": "Data added to collection my_knowledge_base"}
                        }
                    }
                },
                400: {"description": "Data validation failed or collection operation failed"},
                500: {"description": "Internal server error during data processing"}
            }
        )
        async def add_rag_data(
            collection_name: str,
            data_input: RAGDataInput
        ):
            """Add data to RAG collection. Collection name is sanitized for ChromaDB (e.g. spaces → underscores)."""
            collection_name = self.rag_system.sanitize_collection_name(collection_name)
            if not collection_name:
                raise HTTPException(status_code=400, detail="Collection name is required and must be valid (e.g. use letters, numbers, underscores or hyphens only).")
            try:
                success = self.rag_system.add_data_to_collection(collection_name, data_input)
                if success:
                    return {"message": f"Data added to collection {collection_name}"}
                else:
                    raise HTTPException(status_code=400, detail="Failed to add data")
            except Exception as e:
                self.logger.error(f"Error adding data: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/rag/smart-import",
            tags=["RAG"],
            summary="Smart Import",
            description="Intelligently import CSV or JSON files: AI processes and cleans the data, auto-generates collection name, transforms to RAG format, and saves to RAG system.",
            response_model=SmartImportResponse,
            response_description="Smart import results including collection name, processed data, and metadata.",
            responses={
                200: {
                    "description": "Smart import completed successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": True,
                                "collection_name": "customer_data_2024",
                                "collection_description": "Customer data imported and processed with AI",
                                "processed_data": "[{\"id\": 1, \"name\": \"John\", ...}]",
                                "original_record_count": 100,
                                "processed_record_count": 98,
                                "message": "Successfully imported and processed 100 records into collection 'customer_data_2024'",
                                "metadata": {
                                    "llm_provider": "gemini",
                                    "model_used": "gemini-2.5-flash"
                                }
                            }
                        }
                    }
                },
                400: {"description": "Invalid file format or content"},
                500: {"description": "Internal server error during processing"}
            }
        )
        async def smart_import(req: SmartImportRequest) -> SmartImportResponse:
            """
            **Smart Import - AI-Powered Data Import and Processing**
            
            Intelligently imports CSV or JSON files with AI-powered processing:
            
            **Process Flow:**
            1. **Parse File**: Validates and parses CSV or JSON file
            2. **AI Processing**: Uses LLM to clean, abstract, and transform data
               - Removes duplicates and inconsistencies
               - Standardizes formats
               - Extracts key information
               - Creates meaningful summaries
               - Structures for optimal RAG searchability
            3. **Auto-Naming**: AI generates descriptive collection name and description
            4. **RAG Format**: Converts to RAG-optimized JSON format
            5. **Save**: Automatically saves to RAG collection
            
            **Features:**
            - **Intelligent Cleaning**: AI removes noise and standardizes data
            - **Auto-Naming**: AI generates meaningful collection names
            - **RAG Optimization**: Data structured for best search performance
            - **Custom Instructions**: Optional processing instructions for specific needs
            - **Provider Selection**: Choose LLM provider (Gemini, Qwen, Mistral)
            
            **Use Cases:**
            - Import messy CSV/JSON data and get clean, searchable RAG collections
            - Automatically organize and name imported datasets
            - Transform raw data into knowledge base format
            - Clean and standardize data from various sources
            
            **Parameters:**
            - `file_content`: The file content as string (CSV or JSON)
            - `file_format`: "csv" or "json"
            - `llm_provider`: Optional LLM provider (default: system default)
            - `model_name`: Optional model name (default: provider default)
            - `processing_instructions`: Optional custom instructions for AI processing
            - `auto_name`: Whether to auto-generate collection name (default: true)
            
            **Returns:**
            - Collection name and description
            - Processed data in RAG format
            - Record counts (original vs processed)
            - Processing metadata
            """
            try:
                result = await self.rag_system.smart_import(
                    file_content=req.file_content,
                    file_format=req.file_format,
                    llm_provider=req.llm_provider.value if req.llm_provider else None,
                    model_name=req.model_name,
                    processing_instructions=req.processing_instructions,
                    auto_name=req.auto_name,
                )
                return SmartImportResponse(**result)
            except Exception as e:
                self.logger.error(f"Error in smart import: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/rag/collections",
            tags=["RAG"],
            summary="List RAG Collections",
            description="Retrieve a list of all RAG collections in the system with their metadata, document counts, and configuration details.",
            response_description="Array of collection objects with name, document count, and metadata."
        )
        async def list_rag_collections():
            """
            **List All RAG Collections**
            
            Returns comprehensive information about all RAG collections in the system.
            
            **Response Includes:**
            - Collection names
            - Document/record counts per collection
            - Collection metadata
            - Creation and modification information
            
            **Use Cases:**
            - Discover available knowledge bases
            - Monitor collection sizes
            - Check collection health
            - Plan data management operations
            
            **Example Response:**
            ```json
            [
              {
                "name": "product_docs",
                "count": 1250,
                "metadata": {"description": "Product documentation"}
              },
              {
                "name": "faq_knowledge",
                "count": 342,
                "metadata": {"description": "Frequently asked questions"}
              }
            ]
            ```
            """
            try:
                return self.rag_system.list_collections()
            except Exception as e:
                self.logger.error(f"Error listing collections: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/rag/collections/{collection_name}/query",
            tags=["RAG"],
            summary="Query RAG Collection",
            description="Perform semantic search on a RAG collection to find the most relevant documents based on the query. Uses vector similarity search.",
            response_description="Search results containing relevant documents with similarity scores.",
            responses={
                200: {
                    "description": "Query executed successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "results": [
                                    {
                                        "content": "Document content here...",
                                        "metadata": {"source": "doc1"},
                                        "distance": 0.23
                                    }
                                ]
                            }
                        }
                    }
                },
                404: {"description": "Collection not found"},
                500: {"description": "Error during query execution"}
            }
        )
        async def query_rag_collection(collection_name: str, request: RAGQueryRequest):
            """
            **Query RAG Collection with Semantic Search**
            
            Performs vector similarity search on a RAG collection to find documents most relevant to the query.
            
            **How It Works:**
            1. Converts the query text into a vector embedding
            2. Searches the collection for vectors with highest cosine similarity
            3. Returns the top N most relevant documents
            4. Includes similarity scores (lower distance = more relevant)
            
            **Parameters:**
            - `collection_name`: Name of the RAG collection to search
            - `query`: Natural language query string
            - `n_results`: Number of results to return (1-100, default: 5)
            
            **Search Characteristics:**
            - **Semantic Understanding**: Finds documents by meaning, not just keyword matching
            - **Context-Aware**: Understands synonyms, related concepts, and context
            - **Ranked Results**: Results sorted by relevance (distance score)
            - **Metadata Included**: Each result includes source metadata for traceability
            
            **Use Cases:**
            - Finding relevant information in knowledge bases
            - Retrieving context for LLM prompts (RAG)
            - Searching document repositories
            - Answering questions from stored knowledge
            
            **Example Queries:**
            - "How do I reset my password?"
            - "What are the system requirements?"
            - "Explain the authentication process"
            
            **Response Format:**
            Each result includes:
            - `content`: The document text/content
            - `metadata`: Source information and tags
            - `distance`: Similarity score (0.0 = perfect match, higher = less similar)
            """
            try:
                results = self.rag_system.query_collection(collection_name, request.query, request.n_results)
                return {"results": results}
            except Exception as e:
                self.logger.error(f"Error querying collection: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete(
            "/rag/collections/{collection_name}",
            tags=["RAG"],
            summary="Delete RAG Collection",
            description="Permanently delete a RAG collection and all its associated data, embeddings, and metadata. This action cannot be undone.",
            response_description="Confirmation message indicating successful deletion.",
            responses={
                200: {
                    "description": "Collection successfully deleted",
                    "content": {
                        "application/json": {
                            "example": {"message": "Collection product_docs deleted"}
                        }
                    }
                },
                400: {"description": "Failed to delete collection"},
                404: {"description": "Collection not found"},
                500: {"description": "Error during deletion"}
            }
        )
        async def delete_rag_collection(collection_name: str):
            """
            **Delete RAG Collection**
            
            Permanently removes a RAG collection and all associated data from the system.
            
            **What Gets Deleted:**
            - All documents and content in the collection
            - All vector embeddings
            - Collection metadata
            - Index structures
            
            **⚠️ Warning:**
            This operation is **irreversible**. All data in the collection will be permanently lost.
            Consider backing up important data before deletion.
            
            **Use Cases:**
            - Removing outdated or incorrect collections
            - Cleaning up test/development collections
            - Freeing up storage space
            - Rebuilding collections with new data
            
            **Best Practices:**
            - Verify collection name before deletion
            - Export important data before deleting
            - Use this endpoint carefully in production environments
            """
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
        @self.app.post(
            "/agents",
            tags=["Agents"],
            summary="Create AI Agent",
            description="Create a new AI agent with custom configuration including LLM provider, model selection, tools, RAG collections, and behavior settings.",
            response_model=Dict[str, Any],
            response_description="Agent creation response with agent_id and success message.",
            responses={
                200: {
                    "description": "Agent created successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "agent_id": "customer_support_agent",
                                "message": "Agent created successfully"
                            }
                        }
                    }
                },
                500: {"description": "Error during agent creation"}
            }
        )
        async def create_agent(config: AgentConfig):
            """
            **Create New AI Agent**
            
            Creates a fully configured AI agent that can process queries, use tools, and access RAG collections.
            
            **Agent Configuration Options:**
            
            **Basic Settings:**
            - `name`: Unique agent identifier (required)
            - `description`: Human-readable description of agent's purpose
            - `agent_type`: Type of agent (rag, tool, hybrid)
            
            **LLM Configuration:**
            - `llm_provider`: LLM provider (gemini, qwen)
            - `model_name`: Specific model to use (e.g., "gemini-2.5-flash", "qwen3-max")
            - `temperature`: Creativity/randomness (0.0-2.0, default: 0.7)
            - `max_tokens`: Maximum response length (1-32768, default: 8192)
            
            **Capabilities:**
            - `rag_collections`: List of RAG collection names to enable as knowledge sources
            - `tools`: List of tool IDs to enable (e.g., ["web_search", "calculator", "crawler"])
            - `system_prompt`: Custom system prompt defining agent behavior and personality
            
            **Agent Types:**
            - **RAG**: Focused on retrieval-augmented generation from knowledge bases
            - **Tool**: Emphasizes tool usage for actions and external data
            - **Hybrid**: Combines both RAG and tool capabilities
            
            **Use Cases:**
            - Customer support agents with FAQ knowledge
            - Research assistants with document access
            - Task automation agents with tool integration
            - Specialized domain experts with curated knowledge
            
            **Example Configuration:**
            ```json
            {
              "name": "Support Agent",
              "description": "Customer support assistant",
              "agent_type": "hybrid",
              "llm_provider": "gemini",
              "model_name": "gemini-2.5-flash",
              "temperature": 0.7,
              "max_tokens": 4096,
              "rag_collections": ["faq_knowledge", "product_docs"],
              "tools": ["web_search", "calculator"],
              "system_prompt": "You are a helpful customer support agent..."
            }
            ```
            """
            try:
                agent_id = self.agent_manager.create_agent(config)
                return {"agent_id": agent_id, "message": "Agent created successfully"}
            except Exception as e:
                self.logger.error(f"Error creating agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/agents",
            tags=["Agents"],
            summary="List All Agents",
            description="Retrieve a list of all configured agents with their current status, configurations, and capabilities.",
            response_description="Array of agent objects with complete configuration details."
        )
        async def list_agents():
            """
            **List All Configured Agents**
            
            Returns comprehensive information about all agents in the system.
            
            **Response Includes:**
            - Agent ID and name
            - Description and type
            - LLM provider and model configuration
            - Enabled RAG collections
            - Enabled tools
            - Active/inactive status
            - System prompt (if configured)
            
            **Use Cases:**
            - Discover available agents
            - Review agent configurations
            - Monitor agent status
            - Plan agent management operations
            
            **Response Format:**
            Each agent object contains:
            - `id`: Unique agent identifier
            - `name`: Display name
            - `description`: Agent purpose description
            - `agent_type`: Type (rag/tool/hybrid)
            - `model_name`: LLM model being used
            - `is_active`: Whether agent is currently active
            - `rag_collections`: List of enabled knowledge bases
            - `tools`: List of enabled tool IDs
            """
            try:
                return self.agent_manager.list_agents()
            except Exception as e:
                self.logger.error(f"Error listing agents: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/agents/{agent_id}",
            tags=["Agents"],
            summary="Get Agent Details",
            description="Retrieve detailed information about a specific agent including full configuration, status, and runtime information.",
            response_description="Complete agent details with configuration and runtime status.",
            responses={
                200: {"description": "Agent found and returned"},
                404: {"description": "Agent not found"}
            }
        )
        async def get_agent(agent_id: str):
            """
            **Get Agent Details**
            
            Returns comprehensive details about a specific agent.
            
            **Information Included:**
            - Complete configuration (all settings)
            - Current runtime status
            - Active provider and model
            - Enabled capabilities (RAG collections, tools)
            - System prompt configuration
            
            **Use Cases:**
            - Inspect agent configuration
            - Debug agent behavior
            - Verify agent settings
            - Prepare for agent updates
            
            **Response Includes:**
            - Full `config` object with all settings
            - `provider`: Actual LLM provider in use
            - `model`: Actual model being used
            - Runtime status information
            """
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

        @self.app.put(
            "/agents/{agent_id}",
            tags=["Agents"],
            summary="Update Agent Configuration",
            description="Update an existing agent's configuration. The agent will be recreated with new settings, preserving the same agent_id.",
            response_description="Confirmation message indicating successful update.",
            responses={
                200: {
                    "description": "Agent updated successfully",
                    "content": {
                        "application/json": {
                            "example": {"message": "Agent updated successfully"}
                        }
                    }
                },
                400: {"description": "Failed to update agent"},
                404: {"description": "Agent not found"},
                500: {"description": "Error during update"}
            }
        )
        async def update_agent(agent_id: str, config: AgentConfig):
            """
            **Update Agent Configuration**
            
            Updates an existing agent with new configuration settings.
            
            **Update Process:**
            1. Validates new configuration
            2. Removes old agent instance
            3. Creates new agent with updated settings
            4. Preserves the same agent_id
            
            **What Can Be Updated:**
            - LLM provider and model
            - Temperature and max_tokens
            - RAG collections
            - Tools
            - System prompt
            - Agent type
            - Active status
            
            **Important Notes:**
            - Agent is recreated, so any in-flight requests may be affected
            - Agent_id remains the same
            - All configuration fields must be provided (full replacement)
            - Changes take effect immediately
            
            **Use Cases:**
            - Adjusting agent behavior
            - Changing LLM model
            - Adding/removing capabilities
            - Updating system prompts
            - Modifying agent parameters
            """
            try:
                success = self.agent_manager.update_agent(agent_id, config)
                if success:
                    return {"message": "Agent updated successfully"}
                else:
                    raise HTTPException(status_code=400, detail="Failed to update agent")
            except Exception as e:
                self.logger.error(f"Error updating agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete(
            "/agents/{agent_id}",
            tags=["Agents"],
            summary="Delete Agent",
            description="Permanently delete an agent and remove it from the system. This action cannot be undone.",
            response_description="Confirmation message indicating successful deletion.",
            responses={
                200: {
                    "description": "Agent deleted successfully",
                    "content": {
                        "application/json": {
                            "example": {"message": "Agent deleted successfully"}
                        }
                    }
                },
                400: {"description": "Failed to delete agent"},
                404: {"description": "Agent not found"},
                500: {"description": "Error during deletion"}
            }
        )
        async def delete_agent(agent_id: str):
            """
            **Delete Agent**
            
            Permanently removes an agent from the system.
            
            **What Gets Deleted:**
            - Agent configuration
            - Agent instance
            - Agent metadata
            
            **What Is Preserved:**
            - RAG collections (not deleted)
            - Tools (still available)
            - Other agents (unaffected)
            
            **⚠️ Warning:**
            This operation is **irreversible**. The agent and its configuration will be permanently lost.
            
            **Use Cases:**
            - Removing unused or obsolete agents
            - Cleaning up test agents
            - Freeing system resources
            - Reorganizing agent structure
            """
            try:
                success = self.agent_manager.delete_agent(agent_id)
                if success:
                    return {"message": "Agent deleted successfully"}
                else:
                    raise HTTPException(status_code=400, detail="Failed to delete agent")
            except Exception as e:
                self.logger.error(f"Error deleting agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/agents/{agent_id}/run",
            tags=["Agents"],
            summary="Execute Agent",
            description="Execute an agent with a query. The agent will process the query using its configured LLM, tools, and RAG collections, then return a response.",
            response_model=QueryResponse,
            response_description="Agent response with answer, sources, and metadata.",
            responses={
                200: {
                    "description": "Agent executed successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "response": "Based on the knowledge base...",
                                "sources": [],
                                "metadata": {}
                            }
                        }
                    }
                },
                404: {"description": "Agent not found"},
                500: {"description": "Error during agent execution"}
            }
        )
        async def run_agent(agent_id: str, request: QueryRequest):
            """
            **Execute Agent with Query**
            
            Runs an agent to process a query and generate a response.
            
            **Execution Flow:**
            1. Agent receives query and optional context
            2. If RAG collections enabled: Searches knowledge bases for relevant information
            3. If tools enabled: Agent decides which tools to use and executes them
            4. LLM processes query with retrieved context and tool results
            5. Returns final response
            
            **Parameters:**
            - `agent_id`: Identifier of the agent to execute
            - `query`: Natural language query/question
            - `context`: Optional additional context dictionary
            
            **Agent Capabilities:**
            - **RAG Integration**: Automatically searches knowledge bases when relevant
            - **Tool Usage**: Can use enabled tools (web search, calculator, etc.)
            - **Reasoning**: Uses ReAct pattern for multi-step reasoning
            - **Context Awareness**: Incorporates context and retrieved information
            
            **Response Includes:**
            - `response`: The agent's answer
            - `sources`: Source documents used (if RAG was used)
            - `metadata`: Execution metadata (model used, tokens, etc.)
            
            **Use Cases:**
            - Answering questions using knowledge bases
            - Performing tasks with tools
            - Research and information gathering
            - Complex multi-step problem solving
            
            **Example Queries:**
            - "What are the system requirements for Product X?"
            - "Calculate the total cost for 5 units at $25 each"
            - "Search the web for latest news about AI"
            """
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

        @self.app.post(
            "/agents/{agent_id}/run/stream",
            tags=["Agents"],
            summary="Execute Agent with Streaming",
            description="Execute an agent with real-time streaming response. Returns text chunks as they are generated, providing immediate feedback to users.",
            response_description="Streaming text response (text/plain format).",
            responses={
                200: {
                    "description": "Streaming response initiated",
                    "content": {
                        "text/plain": {
                            "example": "The agent is processing...\n\nBased on the knowledge base..."
                        }
                    }
                },
                404: {"description": "Agent not found"},
                500: {"description": "Error during streaming execution"}
            }
        )
        async def run_agent_stream(agent_id: str, request: QueryRequest):
            """
            **Execute Agent with Streaming Response**
            
            Runs an agent with real-time streaming output for better user experience.
            
            **Streaming Benefits:**
            - **Immediate Feedback**: Users see responses as they're generated
            - **Better UX**: No need to wait for complete response
            - **Progress Indication**: Shows when agent is thinking/processing
            - **Lower Perceived Latency**: Feels faster than waiting for full response
            
            **How It Works:**
            1. Agent processes query (may take time for tool calls/RAG)
            2. Once LLM starts generating, text is streamed chunk by chunk
            3. Client receives text in real-time
            4. Stream completes when agent finishes
            
            **Response Format:**
            - Content-Type: `text/plain`
            - Streaming: Server-Sent Events (SSE) compatible
            - Encoding: UTF-8
            
            **Use Cases:**
            - Interactive chat interfaces
            - Real-time assistant applications
            - Long-running queries where immediate feedback is important
            - User-facing applications requiring responsive UX
            
            **Client Implementation:**
            ```javascript
            const response = await fetch('/agents/my_agent/run/stream', {
              method: 'POST',
              body: JSON.stringify({query: "Your question"})
            });
            const reader = response.body.getReader();
            // Read chunks and display in real-time
            ```
            """
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
        @self.app.post(
            "/llm/direct",
            tags=["Direct LLM"],
            summary="Direct LLM Call",
            description="Call an LLM directly without agent orchestration. Supports optional web search tool and custom parameters. Useful for simple queries or testing.",
            response_model=DirectLLMResponse,
            response_description="LLM response with model information and metadata.",
            responses={
                200: {
                    "description": "LLM call successful",
                    "content": {
                        "application/json": {
                            "example": {
                                "response": "The answer to your question...",
                                "model_used": "gemini-2.5-flash",
                                "web_search_used": True,
                                "metadata": {
                                    "temperature": 0.7,
                                    "max_tokens": 8192,
                                    "provider": "gemini"
                                }
                            }
                        }
                    }
                },
                503: {"description": "LLM service unavailable (network/DNS issues)"},
                504: {"description": "LLM request timeout"},
                500: {"description": "Error during LLM call"}
            }
        )
        async def call_llm_direct(request: DirectLLMRequest) -> DirectLLMResponse:
            """
            **Direct LLM Call**
            
            Calls a language model directly without agent orchestration. This is useful for:
            - Simple queries that don't need agent reasoning
            - Testing LLM models and configurations
            - Quick responses without full agent setup
            - Custom use cases requiring direct model access
            
            **Features:**
            - **Model Selection**: Choose any available model by name
            - **Parameter Control**: Set temperature, max_tokens, and system prompt
            - **Optional Web Search**: Enable web search tool for current information
            - **Fast Response**: Bypasses agent overhead for quicker responses
            
            **Request Parameters:**
            - `query`: The question or prompt to send to the LLM
            - `model_name`: Model identifier (e.g., "gemini-2.5-flash", "qwen3-max")
            - `temperature`: Creativity level (0.0-2.0, default: 0.7)
            - `max_tokens`: Maximum response length (1-32768, default: 8192)
            - `use_web_search`: Enable web search tool (default: true)
            - `system_prompt`: Custom system prompt (optional)
            
            **Model Detection:**
            - Models starting with "gemini" → Uses Gemini API
            - Models starting with "qwen" → Uses Qwen API
            - Unknown models → Defaults to Gemini
            
            **Web Search Integration:**
            When enabled, the LLM can search the web for current information and incorporate it into responses.
            
            **Response Includes:**
            - `response`: The LLM's answer
            - `model_used`: Actual model that processed the request
            - `web_search_used`: Whether web search was utilized
            - `metadata`: Request parameters and provider information
            
            **Error Handling:**
            - **503 Service Unavailable**: Network/DNS issues, API unreachable
            - **504 Gateway Timeout**: Request exceeded timeout limit
            - **500 Internal Error**: Other processing errors
            
            **Use Cases:**
            - Quick question answering
            - Model testing and evaluation
            - Simple text generation
            - Web-enhanced queries
            - Custom LLM integrations
            """
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

                    # Build tool names list
                    tool_names_str = ", ".join([t.name for t in tools])
                    
                    # Create ReAct prompt template with tool_names as a variable
                    react_template = system_prompt + f"""

You have access to the following tools:

{{tools}}

IMPORTANT INSTRUCTIONS:
- ALWAYS use the available tools when they can help answer the question
- Do NOT say you cannot do something if you have a tool that can do it
- ONLY use tools that are listed above - do NOT try to use tools that are not in the list
- Available tool names: {tool_names_str}
- Read tool descriptions carefully to understand what each tool can do
- If a tool is not available, explain that you don't have access to that specific tool

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do and which tool to use
Action: the action to take, should be one of [{tool_names_str}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
{{agent_scratchpad}}"""

                    prompt = PromptTemplate(
                        input_variables=["tools", "input", "agent_scratchpad"],
                        template=react_template,
                        partial_variables={"tool_names": tool_names_str}
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
                            response_text = await llm.ainvoke(f"{system_prompt}\n\nQuery: {request.query}")
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
                            llm.ainvoke(f"{system_prompt}\n\nQuery: {request.query}"),
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
        @self.app.get(
            "/tools",
            tags=["Tools"],
            summary="List Available Tools",
            description="Retrieve a comprehensive list of all available tools that can be used by agents, including their configurations, status, and capabilities.",
            response_description="Array of tool objects with complete configuration details."
        )
        async def list_tools():
            """
            **List All Available Tools**
            
            Returns information about all tools available in the system.
            
            **Tool Information Includes:**
            - Tool ID and display name
            - Tool type (web_search, calculator, email, etc.)
            - Description of tool capabilities
            - Configuration settings
            - Active/inactive status
            
            **Available Tool Types:**
            - **Web Search**: Search the internet for current information
            - **Wikipedia**: Search Wikipedia for factual information
            - **Calculator**: Perform mathematical calculations
            - **Email**: Send emails via SMTP
            - **Financial**: Get real-time stock prices and financial data (yfinance - most reliable, free, no API key required)
            - **Crawler**: Crawl websites and extract data
            - **Equalizer**: AI-powered decision making assistance
            - **Custom**: User-defined custom tools
            
            **Use Cases:**
            - Discover available tools for agent configuration
            - Review tool capabilities and configurations
            - Check tool status (active/inactive)
            - Plan tool integration strategies
            
            **Response Format:**
            Each tool object contains:
            - `id`: Unique tool identifier
            - `name`: Display name
            - `tool_type`: Type/category of tool
            - `description`: What the tool does
            - `is_active`: Whether tool is currently enabled
            - `config`: Tool-specific configuration
            """
            try:
                return self.tool_manager.list_tools()
            except Exception as e:
                self.logger.error(f"Error listing tools: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put(
            "/tools/{tool_id}",
            tags=["Tools"],
            summary="Update Tool Configuration",
            description="Update the configuration of a specific tool. Changes take effect immediately and affect all agents using the tool.",
            response_description="Confirmation message indicating successful update.",
            responses={
                200: {
                    "description": "Tool configuration updated successfully",
                    "content": {
                        "application/json": {
                            "example": {"message": "Tool configuration updated successfully"}
                        }
                    }
                },
                400: {"description": "Failed to update tool configuration"},
                404: {"description": "Tool not found"},
                500: {"description": "Error during update"}
            }
        )
        async def update_tool_config(tool_id: str, config: ToolConfig):
            """
            **Update Tool Configuration**
            
            Modifies the configuration of an existing tool.
            
            **What Can Be Updated:**
            - Tool name and description
            - Active/inactive status
            - Tool-specific configuration settings
            - Custom metadata
            
            **Configuration Impact:**
            - Changes apply immediately
            - All agents using the tool are affected
            - Tool behavior may change based on new settings
            
            **Use Cases:**
            - Enabling/disabling tools
            - Updating tool settings (API keys, endpoints, etc.)
            - Modifying tool behavior
            - Adjusting tool metadata
            
            **Important Notes:**
            - Some tools require specific configuration (e.g., email needs SMTP settings)
            - Disabling a tool prevents agents from using it
            - Configuration validation is performed before update
            """
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
        @self.app.get(
            "/models",
            tags=["Models"],
            summary="List Available Models",
            description="Retrieve a list of all available LLM models from configured providers, including their specifications, capabilities, and provider information.",
            response_description="Array of model objects with specifications and provider details."
        )
        async def list_models():
            """
            **List All Available LLM Models**
            
            Returns comprehensive information about all LLM models available in the system.
            
            **Model Information Includes:**
            - Model name/identifier
            - Provider (Gemini, Qwen, etc.)
            - Model description
            - Capabilities and specifications
            
            **Use Cases:**
            - Discover available models for agent configuration
            - Compare model capabilities
            - Select appropriate models for use cases
            - Verify model availability
            
            **Response Format:**
            Each model object contains:
            - `name`: Model identifier (e.g., "gemini-2.5-flash")
            - `provider`: Provider name (e.g., "gemini")
            - `description`: Model description and capabilities
            
            **Model Selection:**
            Use model names from this list when configuring agents or making direct LLM calls.
            """
            try:
                return self.agent_manager.get_available_models()
            except Exception as e:
                self.logger.error(f"Error listing models: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/providers",
            tags=["Models"],
            summary="List LLM Providers",
            description="Retrieve information about all configured LLM providers, their availability, and the default provider setting.",
            response_description="Provider information including available providers and default setting."
        )
        async def list_providers():
            """
            **List LLM Providers**
            
            Returns information about configured LLM providers.
            
            **Response Includes:**
            - `providers`: List of available provider names
            - `default`: Currently configured default provider
            
            **Available Providers:**
            - **Gemini**: Google's Gemini models
            - **Qwen**: Alibaba's Qwen models
            - **Mistral**: Mistral AI models
            
            **Use Cases:**
            - Check which providers are configured
            - Verify default provider setting
            - Discover available provider options
            - Plan provider usage strategy
            
            **Provider Configuration:**
            Providers are configured via environment variables or config file.
            Each provider requires valid API keys to function.
            """
            try:
                return {
                    "providers": self.agent_manager.get_available_providers(),
                    "default": settings.default_llm_provider
                }
            except Exception as e:
                self.logger.error(f"Error listing providers: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # MCP Endpoints
        @self.app.post(
            "/mcp/start",
            tags=["MCP"],
            summary="Start MCP Server",
            description="Start the Model Context Protocol (MCP) server for enhanced AI interactions via WebSocket. The server runs in the background.",
            response_description="Confirmation message indicating server startup initiation.",
            responses={
                200: {
                    "description": "MCP server startup initiated",
                    "content": {
                        "application/json": {
                            "example": {"message": "MCP server starting"}
                        }
                    }
                },
                500: {"description": "Error starting MCP server"}
            }
        )
        async def start_mcp_server(background_tasks: BackgroundTasks):
            """
            **Start Model Context Protocol (MCP) Server**
            
            Initiates the MCP server for WebSocket-based AI interactions.
            
            **MCP Protocol:**
            The Model Context Protocol enables enhanced communication with AI models through:
            - WebSocket connections
            - Structured message protocols
            - Real-time bidirectional communication
            - Enhanced context management
            
            **Server Features:**
            - WebSocket endpoint (default port: 8196)
            - Tool execution support
            - RAG query capabilities
            - Agent execution
            - Health check (ping/pong)
            
            **Server Behavior:**
            - Starts in background (non-blocking)
            - Runs until server shutdown
            - Supports multiple concurrent connections
            - Handles client initialization and capabilities
            
            **Use Cases:**
            - Real-time AI applications
            - Interactive AI interfaces
            - WebSocket-based integrations
            - Enhanced context-aware interactions
            
            **Connection Details:**
            - Protocol: WebSocket
            - Default Port: 8196
            - Message Format: JSON with length prefix
            - Supports multiple clients simultaneously
            """
            try:
                background_tasks.add_task(self.mcp_service.start_server)
                return {"message": "MCP server starting"}
            except Exception as e:
                self.logger.error(f"Error starting MCP server: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Customization Endpoints
        @self.app.post(
            "/customizations",
            tags=["Customizations"],
            summary="Create Customization Profile",
            description="Create a new customization profile with system prompts, optional RAG collection, and LLM configuration. Customizations allow reusable AI behavior templates.",
            response_description="Customization creation response with profile ID.",
            responses={
                200: {
                    "description": "Customization created successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "id": "customer_support_template",
                                "message": "Customization created successfully"
                            }
                        }
                    }
                },
                500: {"description": "Error creating customization"}
            }
        )
        async def create_customization(req: CustomizationCreateRequest):
            """
            **Create Customization Profile**
            
            Creates a reusable customization profile that defines AI behavior, knowledge sources, and model settings.
            
            **Customization Components:**
            - **System Prompt**: Instructions defining AI behavior and personality
            - **RAG Collection**: Optional knowledge base to use as context
            - **LLM Provider**: Optional provider override (defaults to system default)
            - **Model Name**: Optional model override (defaults to provider default)
            - **Description**: Human-readable description of the customization's purpose
            
            **Use Cases:**
            - Creating reusable AI behavior templates
            - Defining specialized AI assistants (support, research, etc.)
            - Setting up domain-specific knowledge bases
            - Standardizing AI interactions across applications
            
            **Profile ID Generation:**
            - Automatically generated from name (URL-friendly)
            - Lowercase, spaces replaced with underscores
            - Unique ID ensures no conflicts
            
            **Example Use Cases:**
            - Customer support template with FAQ knowledge
            - Research assistant with document collection
            - Code review assistant with coding standards
            - Content writer with style guidelines
            
            **Request Fields:**
            - `name`: Profile name (required)
            - `description`: Purpose description (optional)
            - `system_prompt`: Behavior instructions (required)
            - `rag_collection`: Knowledge base name (optional)
            - `llm_provider`: Provider override (optional)
            - `model_name`: Model override (optional)
            - `metadata`: Additional metadata (optional)
            """
            try:
                profile_id = self.customization_manager.create_profile(req)
                return {"id": profile_id, "message": "Customization created successfully"}
            except Exception as e:
                self.logger.error(f"Error creating customization: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/customizations",
            tags=["Customizations"],
            summary="List Customization Profiles",
            description="Retrieve a list of all customization profiles with their configurations, settings, and metadata.",
            response_description="Array of customization profile objects."
        )
        async def list_customizations():
            """
            **List All Customization Profiles**
            
            Returns all customization profiles in the system.
            
            **Response Includes:**
            - Profile ID and name
            - Description and purpose
            - System prompt
            - Associated RAG collection (if any)
            - LLM provider and model overrides (if any)
            - Metadata
            
            **Use Cases:**
            - Discover available customization templates
            - Review customization configurations
            - Plan customization usage
            - Manage customization library
            """
            try:
                return [p.model_dump() for p in self.customization_manager.list_profiles()]
            except Exception as e:
                self.logger.error(f"Error listing customizations: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/customizations/{profile_id}",
            tags=["Customizations"],
            summary="Get Customization Profile",
            description="Retrieve detailed information about a specific customization profile by its ID.",
            response_description="Complete customization profile details.",
            responses={
                200: {"description": "Customization profile found"},
                404: {"description": "Customization profile not found"}
            }
        )
        async def get_customization(profile_id: str):
            """
            **Get Customization Profile Details**
            
            Returns complete information about a specific customization profile.
            
            **Information Included:**
            - Full profile configuration
            - System prompt
            - RAG collection association
            - LLM provider and model settings
            - Metadata
            
            **Use Cases:**
            - Inspect customization configuration
            - Review system prompts
            - Verify settings before use
            - Prepare for updates
            """
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

        @self.app.put(
            "/customizations/{profile_id}",
            tags=["Customizations"],
            summary="Update Customization Profile",
            description="Update an existing customization profile with new settings. All fields can be modified while preserving the profile ID.",
            response_description="Confirmation message indicating successful update.",
            responses={
                200: {
                    "description": "Customization updated successfully",
                    "content": {
                        "application/json": {
                            "example": {"message": "Customization updated successfully"}
                        }
                    }
                },
                404: {"description": "Customization profile not found"},
                500: {"description": "Error during update"}
            }
        )
        async def update_customization(profile_id: str, req: CustomizationCreateRequest):
            """
            **Update Customization Profile**
            
            Modifies an existing customization profile with new configuration.
            
            **What Can Be Updated:**
            - System prompt (behavior instructions)
            - RAG collection association
            - LLM provider and model settings
            - Description and metadata
            - Profile name
            
            **Update Process:**
            - Profile ID remains unchanged
            - All fields are replaced with new values
            - Changes take effect immediately
            - Existing queries using the profile are unaffected
            
            **Use Cases:**
            - Refining AI behavior instructions
            - Changing knowledge base associations
            - Updating model preferences
            - Improving customization templates
            """
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

        @self.app.delete(
            "/customizations/{profile_id}",
            tags=["Customizations"],
            summary="Delete Customization Profile",
            description="Permanently delete a customization profile. This action cannot be undone.",
            response_description="Confirmation message indicating successful deletion.",
            responses={
                200: {
                    "description": "Customization deleted successfully",
                    "content": {
                        "application/json": {
                            "example": {"message": "Customization deleted successfully"}
                        }
                    }
                },
                404: {"description": "Customization profile not found"},
                500: {"description": "Error during deletion"}
            }
        )
        async def delete_customization(profile_id: str):
            """
            **Delete Customization Profile**
            
            Permanently removes a customization profile from the system.
            
            **⚠️ Warning:**
            This operation is **irreversible**. The customization profile and all its settings will be permanently lost.
            
            **What Gets Deleted:**
            - Customization profile configuration
            - System prompt and settings
            - Profile metadata
            
            **What Is Preserved:**
            - RAG collections (not deleted)
            - Other customization profiles
            - System settings
            
            **Use Cases:**
            - Removing obsolete templates
            - Cleaning up test customizations
            - Reorganizing customization library
            """
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

        @self.app.post(
            "/customizations/{profile_id}/query",
            tags=["Customizations"],
            summary="Query Customization Profile",
            description="Execute a query using a customization profile. Combines the profile's system prompt with optional RAG context and generates a response using the configured LLM.",
            response_model=CustomizationQueryResponse,
            response_description="Response generated using the customization profile with metadata about model and RAG usage.",
            responses={
                200: {
                    "description": "Query executed successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "response": "Based on the customization profile...",
                                "profile_id": "customer_support_template",
                                "profile_name": "Customer Support Template",
                                "model_used": "gemini-2.5-flash",
                                "rag_collection_used": "faq_knowledge",
                                "metadata": {
                                    "temperature": 0.7,
                                    "max_tokens": 8192,
                                    "provider": "gemini"
                                }
                            }
                        }
                    }
                },
                404: {"description": "Customization profile not found"},
                500: {"description": "Error during query execution"}
            }
        )
        async def query_customization(profile_id: str, request: CustomizationQueryRequest) -> CustomizationQueryResponse:
            """
            **Query Customization Profile**
            
            Executes a query using a customization profile's configuration.
            
            **Execution Process:**
            1. Loads customization profile configuration
            2. If RAG collection specified: Searches knowledge base for relevant context
            3. Combines system prompt + RAG context + user query
            4. Calls LLM with configured provider/model
            5. Returns formatted response
            
            **Request Parameters:**
            - `profile_id`: Customization profile identifier
            - `query`: User's question or prompt
            - `n_results`: Number of RAG results to include (1-20, default: 3)
            - `temperature`: Optional temperature override (0.0-2.0)
            - `max_tokens`: Optional max tokens override (1-32768)
            
            **Response Includes:**
            - `response`: The AI's answer
            - `profile_id`: Profile used
            - `profile_name`: Profile display name
            - `model_used`: Actual model that processed the request
            - `rag_collection_used`: RAG collection used (if any)
            - `metadata`: Execution parameters and settings
            
            **Use Cases:**
            - Quick queries using predefined templates
            - Consistent AI behavior across applications
            - Domain-specific question answering
            - Template-based AI interactions
            
            **Benefits:**
            - Reusable AI configurations
            - Consistent behavior
            - Easy template management
            - Quick deployment of AI capabilities
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
                elif provider_str == "mistral":
                    provider = LLMProviderType.MISTRAL
                    api_key = settings.mistral_api_key
                    model_name = profile.model_name or settings.mistral_default_model
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
                response_text = await llm.ainvoke(full_prompt)

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
        @self.app.post(
            "/crawler/crawl",
            tags=["Crawler"],
            summary="Crawl Website",
            description="Crawl a website, extract and organize content using AI, and save the processed data to a RAG collection. The AI automatically filters and structures the content.",
            response_model=CrawlerResponse,
            response_description="Crawling results including collection information and extracted data.",
            responses={
                200: {
                    "description": "Website crawled and processed successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": True,
                                "url": "https://example.com",
                                "collection_name": "example_com_data",
                                "collection_description": "Data extracted from example.com",
                                "raw_file": "path/to/raw.json",
                                "extracted_file": "path/to/extracted.json",
                                "extracted_data": {"title": "...", "content": "..."}
                            }
                        }
                    }
                },
                500: {
                    "description": "Error during crawling or processing",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": False,
                                "error": "Failed to crawl website: Connection timeout"
                            }
                        }
                    }
                }
            }
        )
        async def crawl_website(request: CrawlerRequest) -> CrawlerResponse:
            """
            **Crawl Website and Extract Data**
            
            Crawls a website, extracts content, uses AI to organize and filter the data, then saves it to a RAG collection.
            
            **Crawling Process:**
            1. **Web Crawling**: Fetches website content (HTML)
            2. **Content Extraction**: Extracts text, links, headings, and structure
            3. **AI Processing**: Uses LLM to organize, filter, and structure the data
            4. **Noise Removal**: AI removes ads, navigation, and irrelevant content
            5. **RAG Storage**: Saves organized data to specified collection
            
            **Request Parameters:**
            - `url`: Website URL to crawl (required)
            - `use_js`: Whether to execute JavaScript (default: false)
            - `llm_provider`: LLM provider for AI processing (optional, uses default if not specified)
            - `model`: Specific model to use (optional)
            - `collection_name`: Target RAG collection name (auto-generated if not provided)
            - `collection_description`: Collection description (auto-generated if not provided)
            - `follow_links`: Follow links recursively to crawl entire site (default: false)
            - `max_depth`: Maximum depth for recursive crawling, 1-10 (default: 3)
            - `max_pages`: Maximum number of pages to crawl, 1-1000 (default: 50)
            - `same_domain_only`: Only follow links within the same domain (default: true)
            - `headers`: Custom HTTP headers as JSON object (e.g., {"Authorization": "Bearer token"})
            
            **AI Processing:**
            The AI automatically:
            - Identifies main topics and useful information
            - Removes navigation elements, ads, and noise
            - Organizes content into structured format
            - Creates summaries and key points
            - Generates appropriate metadata
            
            **Output Files:**
            - `raw_file`: Raw extracted data (before AI processing)
            - `extracted_file`: AI-organized data (after processing)
            - `extracted_data`: Structured data object
            
            **RAG Collection:**
            - Created automatically if doesn't exist
            - Contains organized, searchable content
            - Ready for semantic search queries
            - Includes metadata and source information
            
            **Use Cases:**
            - Building knowledge bases from websites
            - Extracting documentation for RAG
            - Creating searchable content repositories
            - Automating content ingestion
            - Research and information gathering
            
            **Best Practices:**
            - Start with simple, well-structured websites
            - Use specific collection names for organization
            - Review extracted data quality
            - Handle large sites in multiple crawls
            - Respect robots.txt and rate limits
            
            **Limitations:**
            - JavaScript-heavy sites may require `use_js: true`
            - Very large sites may timeout
            - Some sites may block automated access
            - Processing time depends on content size
            """
            try:
                result = self.crawler_service.crawl_and_save(
                    url=request.url,
                    use_js=request.use_js,
                    llm_provider=request.llm_provider,
                    model=request.model,
                    collection_name=request.collection_name,
                    collection_description=request.collection_description,
                    follow_links=request.follow_links,
                    max_depth=request.max_depth,
                    max_pages=request.max_pages,
                    same_domain_only=request.same_domain_only,
                    headers=request.headers
                )
                
                if result.get("success"):
                    return CrawlerResponse(
                        success=True,
                        url=result["url"],
                        collection_name=result.get("collection_name"),
                        collection_description=result.get("collection_description"),
                        raw_file=result.get("raw_file"),
                        extracted_file=result.get("extracted_file"),
                        extracted_data=result.get("extracted_data"),
                        pages_crawled=result.get("pages_crawled"),
                        total_links_found=result.get("total_links_found")
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

        # Crawler Profile Management Endpoints
        @self.app.get(
            "/crawler/profiles",
            tags=["Crawler"],
            summary="List Crawler Profiles",
            description="List all saved crawler profiles with their configuration (URL, collection name, etc.).",
            response_model=List[Dict[str, Any]],
            response_description="Array of crawler profile objects.",
        )
        async def list_crawler_profiles():
            """List all crawler profiles."""
            try:
                return self.crawler_manager.list_profiles()
            except Exception as e:
                self.logger.error(f"Error listing crawler profiles: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/crawler/profiles/{profile_id}",
            tags=["Crawler"],
            summary="Get Crawler Profile",
            description="Get a specific crawler profile by ID, including URL, use_js, collection_name, max_pages, etc.",
            response_model=CrawlerProfile,
            response_description="Crawler profile object.",
        )
        async def get_crawler_profile(profile_id: str):
            """Get a crawler profile by ID."""
            try:
                profile = self.crawler_manager.get_profile(profile_id)
                if not profile:
                    raise HTTPException(status_code=404, detail="Crawler profile not found")
                return profile
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting crawler profile: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/crawler/profiles",
            tags=["Crawler"],
            summary="Create Crawler Profile",
            description="Create a new crawler profile with URL, optional use_js, collection_name, max_pages, and other options.",
            response_model=CrawlerProfile,
            response_description="Created crawler profile with generated profile_id.",
        )
        async def create_crawler_profile(request: CrawlerCreateRequest):
            """Create a new crawler profile."""
            try:
                return self.crawler_manager.create_profile(request)
            except Exception as e:
                self.logger.error(f"Error creating crawler profile: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put(
            "/crawler/profiles/{profile_id}",
            tags=["Crawler"],
            summary="Update Crawler Profile",
            description="Update an existing crawler profile; all provided fields are updated.",
            response_model=CrawlerProfile,
            response_description="Updated crawler profile.",
        )
        async def update_crawler_profile(profile_id: str, request: CrawlerUpdateRequest):
            """Update a crawler profile."""
            try:
                profile = self.crawler_manager.update_profile(profile_id, request)
                if not profile:
                    raise HTTPException(status_code=404, detail="Crawler profile not found")
                return profile
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error updating crawler profile: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete(
            "/crawler/profiles/{profile_id}",
            tags=["Crawler"],
            summary="Delete Crawler Profile",
            description="Permanently delete a crawler profile by ID.",
            response_model=Dict[str, str],
            response_description="Confirmation message.",
        )
        async def delete_crawler_profile(profile_id: str):
            """Delete a crawler profile."""
            try:
                success = self.crawler_manager.delete_profile(profile_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Crawler profile not found")
                return {"message": "Crawler profile deleted successfully"}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error deleting crawler profile: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/crawler/profiles/{profile_id}/execute",
            tags=["Crawler"],
            summary="Execute Crawler Profile",
            description="Execute a saved crawler profile: crawl the profile's URL, extract content with AI, and save to the profile's RAG collection.",
            response_model=CrawlerResponse,
            response_description="Crawling results (success, collection_name, extracted_data, etc.).",
        )
        async def execute_crawler_profile(profile_id: str):
            """Execute a crawler profile."""
            try:
                profile = self.crawler_manager.get_profile(profile_id)
                if not profile:
                    raise HTTPException(status_code=404, detail="Crawler profile not found")
                
                result = self.crawler_service.crawl_and_save(
                    url=profile.url,
                    use_js=profile.use_js,
                    llm_provider=profile.llm_provider,
                    model=profile.model,
                    collection_name=profile.collection_name,
                    collection_description=profile.collection_description,
                    follow_links=profile.follow_links,
                    max_depth=profile.max_depth,
                    max_pages=profile.max_pages,
                    same_domain_only=profile.same_domain_only,
                    headers=profile.headers
                )
                
                if result.get("success"):
                    return CrawlerResponse(
                        success=True,
                        url=result["url"],
                        collection_name=result.get("collection_name"),
                        collection_description=result.get("collection_description"),
                        raw_file=result.get("raw_file"),
                        extracted_file=result.get("extracted_file"),
                        extracted_data=result.get("extracted_data"),
                        pages_crawled=result.get("pages_crawled"),
                        total_links_found=result.get("total_links_found")
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=result.get("error", "Crawling failed")
                    )
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error executing crawler profile: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=str(e))

        # Database Tools Endpoints
        @self.app.post(
            "/db-tools",
            tags=["Database Tools"],
            summary="Create Database Tool",
            description="Create a new database tool profile with connection configuration and SQL/query statement. The tool will cache query results for the specified TTL period.",
            response_description="Database tool creation response with tool ID.",
            responses={
                200: {
                    "description": "Database tool created successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "tool_id": "customer_db_query",
                                "message": "Database tool created successfully"
                            }
                        }
                    }
                },
                500: {"description": "Error creating database tool"}
            }
        )
        async def create_db_tool(req: DatabaseToolCreateRequest):
            """
            **Create Database Tool Profile**
            
            Creates a new database tool profile that can execute queries against SQL Server, MySQL, SQLite, or MongoDB databases.
            
            **Database Types Supported:**
            - **SQL Server**: Microsoft SQL Server databases
            - **MySQL**: MySQL/MariaDB databases
            - **SQLite**: SQLite file-based databases
            - **MongoDB**: MongoDB NoSQL databases
            
            **Connection Configuration:**
            - `host`: Database server hostname or IP
            - `port`: Database server port
            - `database`: Database name
            - `username`: Database username
            - `password`: Database password
            - `additional_params`: Optional connection parameters (SSL, connection pool, etc.)
            
            **Query Configuration:**
            - **SQL Databases**: Standard SQL SELECT statements
            - **MongoDB**: JSON query format: `{"collection": "users", "query": {...}, "projection": {...}, "limit": 1000}`
            
            **Caching:**
            - Query results are cached for the specified TTL (default: 1 hour)
            - Cache is stored in TinyDB for persistence
            - Cache automatically expires after TTL period
            - Force refresh available via preview endpoint
            
            **Use Cases:**
            - Connecting to production databases for reporting
            - Creating data preview tools
            - Building data dashboards
            - Integrating database data with AI agents
            - Caching expensive queries
            
            **Security Notes:**
            - Passwords are stored in configuration (consider encryption for production)
            - Connection strings are not logged
            - Use read-only database users when possible
            """
            try:
                tool_id = self.db_tools_manager.create_profile(req)
                return {"tool_id": tool_id, "message": "Database tool created successfully"}
            except Exception as e:
                self.logger.error(f"Error creating database tool: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/db-tools",
            tags=["Database Tools"],
            summary="List Database Tools",
            description="Retrieve a list of all database tool profiles with their configurations (passwords are not included in response).",
            response_description="Array of database tool profile objects."
        )
        async def list_db_tools():
            """
            **List All Database Tool Profiles**
            
            Returns all database tool profiles in the system.
            
            **Response Includes:**
            - Tool ID and name
            - Description and database type
            - Connection configuration (password excluded for security)
            - SQL/query statement
            - Active status
            - Cache TTL settings
            - Metadata
            
            **Security:**
            - Passwords are excluded from response for security
            - Only connection parameters are returned
            
            **Use Cases:**
            - Discover available database connections
            - Review tool configurations
            - Manage database tool library
            """
            try:
                profiles = self.db_tools_manager.list_profiles()
                # Remove passwords from response for security
                result = []
                for profile in profiles:
                    profile_dict = profile.model_dump()
                    # Remove password from connection config
                    if "connection_config" in profile_dict and "password" in profile_dict["connection_config"]:
                        profile_dict["connection_config"]["password"] = "***"
                    result.append(profile_dict)
                return result
            except Exception as e:
                self.logger.error(f"Error listing database tools: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/db-tools/{tool_id}",
            tags=["Database Tools"],
            summary="Get Database Tool",
            description="Retrieve detailed information about a specific database tool profile (password excluded for security).",
            response_description="Complete database tool profile details.",
            responses={
                200: {"description": "Database tool found"},
                404: {"description": "Database tool not found"}
            }
        )
        async def get_db_tool(tool_id: str):
            """
            **Get Database Tool Details**
            
            Returns complete information about a specific database tool.
            
            **Information Included:**
            - Full configuration (password excluded)
            - Connection settings
            - SQL/query statement
            - Cache settings
            - Active status
            
            **Security:**
            - Password field is masked for security
            """
            try:
                profile = self.db_tools_manager.get_profile(tool_id)
                if not profile:
                    raise HTTPException(status_code=404, detail="Database tool not found")
                
                profile_dict = profile.model_dump()
                # Remove password for security
                if "connection_config" in profile_dict and "password" in profile_dict["connection_config"]:
                    profile_dict["connection_config"]["password"] = "***"
                
                return profile_dict
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting database tool: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put(
            "/db-tools/{tool_id}",
            tags=["Database Tools"],
            summary="Update Database Tool",
            description="Update an existing database tool profile. Cache will be invalidated for this tool.",
            response_description="Confirmation message indicating successful update.",
            responses={
                200: {
                    "description": "Database tool updated successfully",
                    "content": {
                        "application/json": {
                            "example": {"message": "Database tool updated successfully"}
                        }
                    }
                },
                404: {"description": "Database tool not found"},
                500: {"description": "Error during update"}
            }
        )
        async def update_db_tool(tool_id: str, req: DatabaseToolUpdateRequest):
            """
            **Update Database Tool Profile**
            
            Updates an existing database tool with new configuration.
            
            **Update Process:**
            - Cache is automatically invalidated
            - New queries will use updated configuration
            - Tool ID remains unchanged
            
            **What Can Be Updated:**
            - Connection configuration
            - SQL/query statement
            - Cache TTL
            - Active status
            - Description and metadata
            """
            try:
                success = self.db_tools_manager.update_profile(tool_id, req)
                if success:
                    return {"message": "Database tool updated successfully"}
                raise HTTPException(status_code=404, detail="Database tool not found")
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error updating database tool: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete(
            "/db-tools/{tool_id}",
            tags=["Database Tools"],
            summary="Delete Database Tool",
            description="Permanently delete a database tool profile and its cached data.",
            response_description="Confirmation message indicating successful deletion.",
            responses={
                200: {
                    "description": "Database tool deleted successfully",
                    "content": {
                        "application/json": {
                            "example": {"message": "Database tool deleted successfully"}
                        }
                    }
                },
                404: {"description": "Database tool not found"},
                500: {"description": "Error during deletion"}
            }
        )
        async def delete_db_tool(tool_id: str):
            """
            **Delete Database Tool Profile**
            
            Permanently removes a database tool and its cached data.
            
            **What Gets Deleted:**
            - Tool configuration
            - Cached query results
            - Tool metadata
            
            **⚠️ Warning:**
            This operation is **irreversible**.
            """
            try:
                success = self.db_tools_manager.delete_profile(tool_id)
                if success:
                    return {"message": "Database tool deleted successfully"}
                raise HTTPException(status_code=404, detail="Database tool not found")
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error deleting database tool: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/db-tools/{tool_id}/preview",
            tags=["Database Tools"],
            summary="Preview Database Query Results",
            description="Execute the database query and return the first 10 rows of results. Uses cache if available and valid, otherwise executes query and caches results.",
            response_model=DatabaseToolPreviewResponse,
            response_description="Preview data with first 10 rows, columns, and cache information.",
            responses={
                200: {
                    "description": "Preview data retrieved successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "tool_id": "customer_db_query",
                                "tool_name": "Customer Database Query",
                                "columns": ["id", "name", "email", "created_at"],
                                "rows": [
                                    [1, "John Doe", "john@example.com", "2024-01-01"],
                                    [2, "Jane Smith", "jane@example.com", "2024-01-02"]
                                ],
                                "total_rows": 150,
                                "cached": True,
                                "cache_expires_at": "2024-01-15T10:30:00",
                                "metadata": {}
                            }
                        }
                    }
                },
                404: {"description": "Database tool not found"},
                400: {"description": "Database tool is not active or query execution failed"},
                500: {"description": "Error executing query or retrieving preview"}
            }
        )
        async def preview_db_tool(tool_id: str, force_refresh: bool = False):
            """
            **Preview Database Query Results**
            
            Executes the database query and returns the first 10 rows for preview.
            
            **Query Execution:**
            1. Checks cache first (if `force_refresh=False`)
            2. If cache valid: Returns cached data
            3. If cache expired/missing: Executes query
            4. Stores results in cache with TTL
            5. Returns first 10 rows
            
            **Query Parameters:**
            - `tool_id`: Database tool identifier
            - `force_refresh`: Force query execution, bypassing cache (default: false)
            
            **Response Includes:**
            - `columns`: Column names from query result
            - `rows`: First 10 rows of data (list of lists)
            - `total_rows`: Total number of rows in result set
            - `cached`: Whether data came from cache
            - `cache_expires_at`: When cache expires (ISO format)
            
            **Database-Specific Notes:**
            
            **SQL Server/MySQL/SQLite:**
            - Use standard SQL SELECT statements
            - Example: `SELECT id, name, email FROM users WHERE active = 1`
            - For SQLite: Provide database file path in 'database' field (e.g., '/path/to/database.db')
            
            **MongoDB:**
            - Use JSON query format:
            ```json
            {
              "collection": "users",
              "query": {"active": true},
              "projection": {"_id": 1, "name": 1, "email": 1},
              "limit": 1000
            }
            ```
            
            **Caching Behavior:**
            - Results cached for configured TTL (default: 1 hour)
            - Cache stored in TinyDB for persistence
            - Automatic cache expiration
            - Force refresh available via `force_refresh=true`
            
            **Use Cases:**
            - Preview query results before using in production
            - Verify query correctness
            - Check data structure and content
            - Monitor cached data freshness
            - Debug query issues
            
            **Error Handling:**
            - Connection errors: Returns 400 with error details
            - Query syntax errors: Returns 400 with error message
            - Database unavailable: Returns 500 with connection error
            """
            try:
                result = self.db_tools_manager.preview_data(tool_id, force_refresh=force_refresh)
                return DatabaseToolPreviewResponse(**result)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                self.logger.error(f"Error previewing database tool {tool_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/db-tools/{tool_id}/execute",
            tags=["Database Tools"],
            summary="Execute Database Query with Dynamic SQL",
            description="Execute database query with optional dynamic SQL input. If allow_dynamic_sql is enabled, the input will be combined with preset_sql_statement (as WHERE condition) or used as full SQL if preset is empty.",
            response_model=DatabaseToolPreviewResponse,
            response_description="Query execution result with all rows, columns, and metadata.",
            responses={
                200: {
                    "description": "Query executed successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "tool_id": "customer_db_query",
                                "tool_name": "Customer Database Query",
                                "columns": ["id", "name", "email"],
                                "rows": [[1, "John Doe", "john@example.com"]],
                                "total_rows": 1,
                                "cached": False,
                                "cache_expires_at": None,
                                "metadata": {}
                            }
                        }
                    }
                },
                404: {"description": "Database tool not found"},
                400: {"description": "Database tool is not active or query execution failed"},
                500: {"description": "Error executing query"}
            }
        )
        async def execute_db_tool(tool_id: str, request: Optional[DatabaseToolExecuteRequest] = None):
            """
            **Execute Database Query with Dynamic SQL**
            
            Executes a database query with optional dynamic SQL input.
            
            **Dynamic SQL Behavior:**
            - If `allow_dynamic_sql` is enabled and `sql_input` is provided:
              - If `preset_sql_statement` exists: `sql_input` is appended as WHERE condition
              - If `preset_sql_statement` is empty: `sql_input` is used as full SQL statement
            - If `allow_dynamic_sql` is disabled: Uses the configured `sql_statement`
            
            **Request Body:**
            - `sql_input` (optional): Dynamic SQL input (WHERE condition or full SQL statement)
            
            **SQL Combination Examples:**
            
            **With Preset SQL:**
            - Preset: `SELECT id, name, email FROM users`
            - Input: `active = 1 AND created_at > '2024-01-01'`
            - Result: `SELECT id, name, email FROM users WHERE active = 1 AND created_at > '2024-01-01'`
            
            **Full SQL Input:**
            - Preset: (empty)
            - Input: `SELECT * FROM orders WHERE status = 'pending'`
            - Result: `SELECT * FROM orders WHERE status = 'pending'`
            
            **Use Cases:**
            - Execute queries with dynamic WHERE conditions
            - Use SQL statements generated from previous flow steps
            - Build queries programmatically
            - Test different query variations
            
            **Note:** Dynamic queries are not cached to ensure fresh results.
            """
            try:
                sql_input = request.sql_input if request else None
                result = self.db_tools_manager.execute_query(tool_id, sql_input=sql_input, force_refresh=True)
                return DatabaseToolPreviewResponse(**result)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                self.logger.error(f"Error executing database tool {tool_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Request Tools Endpoints
        @self.app.post(
            "/request-tools",
            tags=["Request Tools"],
            summary="Create Request Configuration",
            description="Create a new request configuration for HTTP API calls or internal service calls. The configuration will be saved to TinyDB.",
            response_description="Request creation response with request ID.",
            responses={
                200: {
                    "description": "Request configuration created successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "request_id": "get_user_data",
                                "message": "Request configuration created successfully"
                            }
                        }
                    }
                },
                400: {"description": "Invalid request configuration or duplicate name"},
                500: {"description": "Error creating request configuration"}
            }
        )
        async def create_request_tool(req: RequestCreateRequest):
            """
            **Create Request Configuration**
            
            Creates a new request configuration that can execute HTTP API calls or internal service calls.
            
            **Request Types:**
            - **HTTP**: External HTTP API requests (GET, POST, PUT, DELETE, etc.)
            - **Internal**: Internal service calls within the project
            
            **Configuration Fields:**
            - `name`: Unique request name/task identifier (required, must be unique)
            - `description`: Optional description
            - `request_type`: "http" or "internal"
            - `method`: HTTP method (required for HTTP requests)
            - `url`: HTTP URL (required for HTTP requests)
            - `endpoint`: Internal endpoint (required for internal requests)
            - `headers`: HTTP headers (key-value pairs)
            - `params`: URL query parameters
            - `body`: Request body (string or JSON object)
            - `timeout`: Request timeout in seconds (1-300, default: 30)
            
            **Storage:**
            - Configuration saved to TinyDB
            - Last response automatically saved after execution
            - Responses overwrite previous results
            
            **Use Cases:**
            - Configure API endpoints for testing
            - Set up recurring API calls
            - Create internal service call templates
            - Monitor external API responses
            """
            try:
                request_id = self.request_tools_manager.create_profile(req)
                return {"request_id": request_id, "message": "Request configuration created successfully"}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                self.logger.error(f"Error creating request tool: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/request-tools",
            tags=["Request Tools"],
            summary="List Request Configurations",
            description="Retrieve a list of all request configurations with their settings and last execution status.",
            response_description="Array of request configuration objects."
        )
        async def list_request_tools():
            """
            **List All Request Configurations**
            
            Returns all request configurations in the system.
            
            **Response Includes:**
            - Request ID and name
            - Description and type
            - Configuration details
            - Last response data
            - Last execution timestamp
            """
            try:
                profiles = self.request_tools_manager.list_profiles()
                return [profile.model_dump() for profile in profiles]
            except Exception as e:
                self.logger.error(f"Error listing request tools: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/request-tools/{request_id}",
            tags=["Request Tools"],
            summary="Get Request Configuration",
            description="Retrieve detailed information about a specific request configuration including last response.",
            response_description="Complete request configuration details.",
            responses={
                200: {"description": "Request configuration found"},
                404: {"description": "Request configuration not found"}
            }
        )
        async def get_request_tool(request_id: str):
            """Get request configuration details including last response."""
            try:
                profile = self.request_tools_manager.get_profile(request_id)
                if not profile:
                    raise HTTPException(status_code=404, detail="Request configuration not found")
                return profile.model_dump()
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting request tool: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put(
            "/request-tools/{request_id}",
            tags=["Request Tools"],
            summary="Update Request Configuration",
            description="Update an existing request configuration. Last response is preserved.",
            response_description="Confirmation message indicating successful update.",
            responses={
                200: {
                    "description": "Request configuration updated successfully",
                    "content": {
                        "application/json": {
                            "example": {"message": "Request configuration updated successfully"}
                        }
                    }
                },
                400: {"description": "Invalid configuration or duplicate name"},
                404: {"description": "Request configuration not found"},
                500: {"description": "Error during update"}
            }
        )
        async def update_request_tool(request_id: str, req: RequestUpdateRequest):
            """
            **Update Request Configuration**
            
            Updates an existing request configuration.
            
            **Update Process:**
            - Last response is preserved
            - Configuration is updated
            - Request ID remains unchanged
            
            **What Can Be Updated:**
            - All configuration fields
            - Request type, method, URL/endpoint
            - Headers, params, body
            - Timeout settings
            """
            try:
                success = self.request_tools_manager.update_profile(request_id, req)
                if success:
                    return {"message": "Request configuration updated successfully"}
                raise HTTPException(status_code=404, detail="Request configuration not found")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error updating request tool: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete(
            "/request-tools/{request_id}",
            tags=["Request Tools"],
            summary="Delete Request Configuration",
            description="Permanently delete a request configuration and its saved response.",
            response_description="Confirmation message indicating successful deletion.",
            responses={
                200: {
                    "description": "Request configuration deleted successfully",
                    "content": {
                        "application/json": {
                            "example": {"message": "Request configuration deleted successfully"}
                        }
                    }
                },
                404: {"description": "Request configuration not found"},
                500: {"description": "Error during deletion"}
            }
        )
        async def delete_request_tool(request_id: str):
            """
            **Delete Request Configuration**
            
            Permanently removes a request configuration and its saved response.
            
            **⚠️ Warning:**
            This operation is **irreversible**.
            """
            try:
                success = self.request_tools_manager.delete_profile(request_id)
                if success:
                    return {"message": "Request configuration deleted successfully"}
                raise HTTPException(status_code=404, detail="Request configuration not found")
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error deleting request tool: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/request-tools/{request_id}/execute",
            tags=["Request Tools"],
            summary="Execute Request",
            description="Execute a configured request (HTTP or internal) and save the response to TinyDB. The response will overwrite the previous result.",
            response_model=RequestExecuteResponse,
            response_description="Execution result with response data and metadata.",
            responses={
                200: {
                    "description": "Request executed successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "request_id": "get_user_data",
                                "request_name": "Get User Data",
                                "success": True,
                                "status_code": 200,
                                "response_data": {"users": []},
                                "response_headers": {},
                                "execution_time": 0.234,
                                "error": None,
                                "executed_at": "2024-01-15T10:30:00",
                                "metadata": {}
                            }
                        }
                    }
                },
                404: {"description": "Request configuration not found"},
                500: {"description": "Error executing request"}
            }
        )
        async def execute_request_tool(request_id: str):
            """
            **Execute Request**
            
            Executes a configured request and saves the response.
            
            **Execution Process:**
            1. Loads request configuration
            2. Executes HTTP or internal request
            3. Saves response to TinyDB (overwrites previous)
            4. Updates last execution timestamp
            5. Returns execution result
            
            **Response Storage:**
            - Response saved automatically
            - Previous response is overwritten
            - Includes status code, data, headers, execution time
            - Error information if request failed
            
            **Request Types:**
            - **HTTP**: Makes external API call
            - **Internal**: Calls internal service endpoint (requires routing)
            
            **Use Cases:**
            - Test API endpoints
            - Monitor external services
            - Execute recurring requests
            - Debug API integrations
            """
            try:
                result = self.request_tools_manager.execute_request(request_id)
                return RequestExecuteResponse(**result)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                self.logger.error(f"Error executing request tool {request_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Dialogue Endpoints
        @self.app.post(
            "/dialogues",
            tags=["Dialogues"],
            summary="Create Dialogue Profile",
            description="Create a new dialogue profile for multi-turn conversations. Similar to Customization but supports back-and-forth dialogue with a maximum number of turns (default: 5). The dialogue allows the AI to ask follow-up questions if more context is needed before providing a final answer.",
            response_description="Dialogue creation response with dialogue ID.",
            responses={
                200: {
                    "description": "Dialogue profile created successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "id": "customer_support_dialogue",
                                "message": "Dialogue created successfully"
                            }
                        }
                    }
                },
                400: {"description": "Invalid request data"},
                500: {"description": "Error creating dialogue profile"}
            }
        )
        async def create_dialogue(req: DialogueCreateRequest):
            """
            **Create Dialogue Profile**
            
            Creates a new dialogue profile that enables multi-turn conversations with AI.
            
            **Key Features:**
            - System prompt configuration for AI behavior
            - Optional RAG collection integration for context
            - LLM provider and model selection
            - Configurable maximum conversation turns (1-10, default: 5)
            - Multi-turn conversation support with automatic continuation detection
            
            **Request Parameters:**
            - `name`: Unique name for the dialogue profile
            - `description`: Optional description of the dialogue's purpose
            - `system_prompt`: Instructions for how the AI should behave
            - `rag_collection`: Optional RAG collection name for context
            - `llm_provider`: Optional LLM provider override (gemini, qwen, mistral)
            - `model_name`: Optional model name override
            - `max_turns`: Maximum conversation turns (1-10, default: 5)
            
            **Use Cases:**
            - Customer support dialogues that gather information before providing solutions
            - Form-filling assistants that ask clarifying questions
            - Consultation dialogues that need multiple exchanges
            - Any scenario requiring back-and-forth conversation
            
            **Example:**
            Create a customer support dialogue that asks for order number, issue type, and description before providing help.
            """
            try:
                dialogue_id = self.dialogue_manager.create_profile(req)
                return {"id": dialogue_id, "message": "Dialogue created successfully"}
            except Exception as e:
                self.logger.error(f"Error creating dialogue: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/dialogues",
            tags=["Dialogues"],
            summary="List All Dialogues",
            description="Retrieve a comprehensive list of all dialogue profiles configured in the system, including their settings, configurations, and metadata.",
            response_description="Array of dialogue profile objects with complete configuration details.",
            responses={
                200: {
                    "description": "List of dialogue profiles retrieved successfully",
                    "content": {
                        "application/json": {
                            "example": [
                                {
                                    "id": "customer_support_dialogue",
                                    "name": "Customer Support Dialogue",
                                    "description": "Multi-turn customer support assistant",
                                    "system_prompt": "You are a helpful customer support agent...",
                                    "rag_collection": "support_knowledge",
                                    "llm_provider": "gemini",
                                    "model_name": "gemini-2.5-flash",
                                    "max_turns": 5,
                                    "metadata": {}
                                }
                            ]
                        }
                    }
                },
                500: {"description": "Error retrieving dialogue profiles"}
            }
        )
        async def list_dialogues():
            """
            **List All Dialogue Profiles**
            
            Returns information about all dialogue profiles available in the system.
            
            **Response Includes:**
            - Profile ID and name
            - System prompt configuration
            - RAG collection associations
            - LLM provider and model settings
            - Maximum turns configuration
            - Metadata and descriptions
            
            **Use Cases:**
            - Discover available dialogue profiles
            - Review dialogue configurations
            - Plan dialogue usage strategies
            - Check dialogue status and settings
            """
            try:
                return [p.model_dump() for p in self.dialogue_manager.list_profiles()]
            except Exception as e:
                self.logger.error(f"Error listing dialogues: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/dialogues/{dialogue_id}",
            tags=["Dialogues"],
            summary="Get Dialogue Profile",
            description="Retrieve detailed information about a specific dialogue profile, including its complete configuration, system prompt, and settings.",
            response_description="Dialogue profile object with all configuration details.",
            responses={
                200: {
                    "description": "Dialogue profile retrieved successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "id": "customer_support_dialogue",
                                "name": "Customer Support Dialogue",
                                "description": "Multi-turn customer support assistant",
                                "system_prompt": "You are a helpful customer support agent...",
                                "rag_collection": "support_knowledge",
                                "llm_provider": "gemini",
                                "model_name": "gemini-2.5-flash",
                                "max_turns": 5,
                                "metadata": {}
                            }
                        }
                    }
                },
                404: {"description": "Dialogue profile not found"},
                500: {"description": "Error retrieving dialogue profile"}
            }
        )
        async def get_dialogue(dialogue_id: str):
            """
            **Get Dialogue Profile Details**
            
            Returns complete information about a specific dialogue profile.
            
            **Profile Information Includes:**
            - Profile ID and display name
            - System prompt configuration
            - RAG collection association (if any)
            - LLM provider and model settings
            - Maximum conversation turns
            - Description and metadata
            
            **Use Cases:**
            - Review dialogue configuration before use
            - Verify dialogue settings
            - Understand dialogue behavior and capabilities
            - Check dialogue profile details
            """
            try:
                dialogue = self.dialogue_manager.get_profile(dialogue_id)
                if not dialogue:
                    raise HTTPException(status_code=404, detail="Dialogue not found")
                return dialogue.model_dump()
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting dialogue {dialogue_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put(
            "/dialogues/{dialogue_id}",
            tags=["Dialogues"],
            summary="Update Dialogue Profile",
            description="Update an existing dialogue profile's configuration, including system prompt, RAG collection, LLM settings, and maximum turns.",
            response_description="Update confirmation message.",
            responses={
                200: {
                    "description": "Dialogue profile updated successfully",
                    "content": {
                        "application/json": {
                            "example": {"message": "Dialogue updated successfully"}
                        }
                    }
                },
                404: {"description": "Dialogue profile not found"},
                500: {"description": "Error updating dialogue profile"}
            }
        )
        async def update_dialogue(dialogue_id: str, req: DialogueUpdateRequest):
            """
            **Update Dialogue Profile**
            
            Modifies an existing dialogue profile's configuration.
            
            **Updateable Fields:**
            - Profile name and description
            - System prompt/instructions
            - RAG collection association
            - LLM provider and model selection
            - Maximum conversation turns
            - Metadata
            
            **Update Behavior:**
            - All provided fields will replace existing values
            - Active conversations using this profile are not affected
            - Changes take effect immediately for new conversations
            
            **Use Cases:**
            - Refine dialogue behavior by updating system prompt
            - Change LLM provider or model
            - Adjust maximum turns based on usage patterns
            - Update RAG collection for better context
            """
            try:
                success = self.dialogue_manager.update_profile(dialogue_id, req)
                if success:
                    return {"message": "Dialogue updated successfully"}
                else:
                    raise HTTPException(status_code=404, detail="Dialogue not found")
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error updating dialogue {dialogue_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete(
            "/dialogues/{dialogue_id}",
            tags=["Dialogues"],
            summary="Delete Dialogue Profile",
            description="Permanently delete a dialogue profile and clean up any associated active conversations. This action cannot be undone.",
            response_description="Deletion confirmation message.",
            responses={
                200: {
                    "description": "Dialogue profile deleted successfully",
                    "content": {
                        "application/json": {
                            "example": {"message": "Dialogue deleted successfully"}
                        }
                    }
                },
                404: {"description": "Dialogue profile not found"},
                500: {"description": "Error deleting dialogue profile"}
            }
        )
        async def delete_dialogue(dialogue_id: str):
            """
            **Delete Dialogue Profile**
            
            Permanently removes a dialogue profile from the system.
            
            **Deletion Effects:**
            - Dialogue profile is removed from storage
            - All active conversations for this profile are terminated
            - Profile configuration is permanently deleted
            - Cannot be undone
            
            **Use Cases:**
            - Remove obsolete dialogue profiles
            - Clean up test or unused dialogues
            - Remove dialogues that are no longer needed
            
            **Warning:** This action is permanent and will terminate any active conversations using this profile.
            """
            try:
                success = self.dialogue_manager.delete_profile(dialogue_id)
                if success:
                    return {"message": "Dialogue deleted successfully"}
                else:
                    raise HTTPException(status_code=404, detail="Dialogue not found")
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error deleting dialogue {dialogue_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/dialogues/{dialogue_id}/start",
            tags=["Dialogues"],
            summary="Start Dialogue Conversation",
            description="Start a new multi-turn dialogue conversation with an initial user message. The AI will process the message, optionally use RAG context if configured, and respond. The response may indicate that more information is needed (needs_more_info=true) or provide a final answer (is_complete=true).",
            response_model=DialogueResponse,
            response_description="AI response with conversation ID, turn number, completion status, and full conversation history.",
            responses={
                200: {
                    "description": "Dialogue conversation started successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                                "turn_number": 1,
                                "max_turns": 5,
                                "response": "I'd be happy to help! To assist you better, could you please provide your order number?",
                                "needs_more_info": True,
                                "is_complete": False,
                                "profile_id": "customer_support_dialogue",
                                "profile_name": "Customer Support Dialogue",
                                "model_used": "gemini-2.5-flash",
                                "rag_collection_used": "support_knowledge",
                                "conversation_history": [
                                    {"role": "user", "content": "I need help with my order", "timestamp": "2024-01-15T10:30:00"},
                                    {"role": "assistant", "content": "I'd be happy to help! To assist you better, could you please provide your order number?", "timestamp": "2024-01-15T10:30:05"}
                                ],
                                "metadata": {
                                    "temperature": 0.7,
                                    "max_tokens": 8192,
                                    "provider": "gemini"
                                }
                            }
                        }
                    }
                },
                404: {"description": "Dialogue profile not found"},
                500: {"description": "Error starting dialogue conversation"}
            }
        )
        async def start_dialogue(dialogue_id: str, request: DialogueStartRequest) -> DialogueResponse:
            """
            **Start Dialogue Conversation**
            
            Initiates a new multi-turn dialogue conversation using a dialogue profile.
            
            **Execution Process:**
            1. Loads dialogue profile configuration
            2. If RAG collection specified: Searches knowledge base for relevant context
            3. Creates new conversation session with unique conversation_id
            4. Combines system prompt + RAG context + user message
            5. Calls LLM with configured provider/model
            6. Analyzes response to determine if more information is needed
            7. Returns response with conversation state
            
            **Request Parameters:**
            - `dialogue_id`: Dialogue profile identifier
            - `initial_message`: User's first message to start the conversation
            - `n_results`: Number of RAG results to include (1-20, default: 3)
            - `temperature`: Optional temperature override (0.0-2.0)
            - `max_tokens`: Optional max tokens override (1-32768)
            
            **Response Fields:**
            - `conversation_id`: Unique ID for this conversation (use for continue endpoint)
            - `turn_number`: Current turn number (starts at 1)
            - `max_turns`: Maximum turns allowed for this dialogue
            - `response`: AI's response text
            - `needs_more_info`: Whether AI is asking for more information
            - `is_complete`: Whether dialogue is complete (final answer provided)
            - `conversation_history`: Full conversation history up to this point
            
            **Conversation Flow:**
            - If `needs_more_info=true`: Use `/continue` endpoint with conversation_id
            - If `is_complete=true`: Conversation is finished, no further turns needed
            - Maximum turns enforced: conversation ends at max_turns even if more info needed
            
            **Use Cases:**
            - Start customer support conversations
            - Begin form-filling dialogues
            - Initiate consultation sessions
            - Start any multi-turn AI interaction
            """
            try:
                profile = self.dialogue_manager.get_profile(dialogue_id)
                if not profile:
                    raise HTTPException(status_code=404, detail="Dialogue not found")

                # Determine provider/model (same logic as customization)
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
                elif provider_str == "mistral":
                    provider = LLMProviderType.MISTRAL
                    api_key = settings.mistral_api_key
                    model_name = profile.model_name or settings.mistral_default_model
                else:
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
                        request.initial_message,
                        request.n_results,
                    )
                    if results:
                        context = "\n\n".join(r["content"] for r in results[: request.n_results])

                # Create tools for this dialogue
                tools = []
                
                # Add DB tools
                if profile.db_tools and self.db_tools_manager:
                    for tool_id in profile.db_tools:
                        db_profile = self.db_tools_manager.get_profile(tool_id)
                        if db_profile and db_profile.is_active:
                            from langchain.tools import Tool
                            
                            def create_db_tool_func(t_id: str):
                                def db_tool_func(query_input: str) -> str:
                                    try:
                                        # If dynamic SQL is enabled, use the input as SQL
                                        if db_profile.allow_dynamic_sql:
                                            result = self.db_tools_manager.execute_query(t_id, sql_input=query_input, force_refresh=True)
                                        else:
                                            result = self.db_tools_manager.execute_query(t_id, force_refresh=True)
                                        
                                        # Format result as string
                                        if result.get("rows"):
                                            rows_str = "\n".join([str(row) for row in result["rows"][:10]])  # Limit to 10 rows
                                            return f"Query executed successfully. Found {result.get('total_rows', len(result.get('rows', [])))} rows.\nColumns: {', '.join(result.get('columns', []))}\nSample rows:\n{rows_str}"
                                        else:
                                            return f"Query executed successfully but returned no rows."
                                    except Exception as e:
                                        return f"Error executing database query: {str(e)}"
                                return db_tool_func
                            
                            tool = Tool(
                                name=f"DB_{tool_id}",
                                func=create_db_tool_func(tool_id),
                                description=f"{db_profile.description or db_profile.name}. Execute database query: {db_profile.sql_statement[:100]}..."
                            )
                            tools.append(tool)
                
                # Add Request tools
                if profile.request_tools and self.request_tools_manager:
                    for tool_id in profile.request_tools:
                        req_profile = self.request_tools_manager.get_profile(tool_id)
                        if req_profile and req_profile.is_active:
                            from langchain.tools import Tool
                            
                            def create_request_tool_func(r_id: str):
                                def request_tool_func(input_data: str = "") -> str:
                                    try:
                                        result = self.request_tools_manager.execute_request(r_id)
                                        if result.get("success"):
                                            return f"Request executed successfully. Status: {result.get('status_code')}. Response: {str(result.get('response_data', {}))[:500]}"
                                        else:
                                            return f"Request failed: {result.get('error', 'Unknown error')}"
                                    except Exception as e:
                                        return f"Error executing request: {str(e)}"
                                return request_tool_func
                            
                            tool = Tool(
                                name=f"Request_{tool_id}",
                                func=create_request_tool_func(tool_id),
                                description=f"{req_profile.description or req_profile.name}. Execute HTTP request: {req_profile.method} {req_profile.url}"
                            )
                            tools.append(tool)

                # Create conversation
                conversation_id = self.dialogue_manager._create_conversation(
                    dialogue_id, request.initial_message, turn_number=1
                )

                # Build messages for conversation
                messages = [
                    {"role": "system", "content": profile.system_prompt},
                    {"role": "user", "content": request.initial_message}
                ]
                if context:
                    messages.insert(1, {"role": "system", "content": f"Context (from knowledge base '{rag_used}'):\n{context}"})

                # Get AI response - use agent executor if tools are available, otherwise direct LLM call
                if tools:
                    from langchain.agents import AgentExecutor, create_react_agent
                    from langchain.prompts import PromptTemplate
                    
                    # Build tool names list for the prompt
                    tool_names_str = ", ".join([t.name for t in tools])
                    
                    try:
                        # Escape curly braces in system prompt to prevent LangChain from treating them as template variables
                        # We need to escape { and } by doubling them, but preserve our actual template variables
                        system_instruction = profile.system_prompt or "You are a helpful AI assistant."
                        # Protect our template variables with placeholders
                        protected_vars = {
                            "{tools}": "___TEMPLATE_VAR_TOOLS___",
                            "{tool_names}": "___TEMPLATE_VAR_TOOL_NAMES___",
                            "{input}": "___TEMPLATE_VAR_INPUT___",
                            "{agent_scratchpad}": "___TEMPLATE_VAR_SCRATCHPAD___"
                        }
                        # Replace protected variables with placeholders
                        for var, placeholder in protected_vars.items():
                            system_instruction = system_instruction.replace(var, placeholder)
                        # Escape all remaining curly braces
                        system_instruction = system_instruction.replace("{", "{{").replace("}", "}}")
                        # Restore our template variables
                        for var, placeholder in protected_vars.items():
                            system_instruction = system_instruction.replace(placeholder, var)
                        
                        system_instruction += " IMPORTANT: You have access to tools that can help you. ALWAYS use the available tools when needed."
                        
                        react_template = system_instruction + """

You have access to the following tools:

{tools}

IMPORTANT INSTRUCTIONS:
- ALWAYS use the available tools when they can help answer the question
- Do NOT say you cannot do something if you have a tool that can do it
- ONLY use tools that are listed above - do NOT try to use tools that are not in the list
- Available tool names: {tool_names}
- Read tool descriptions carefully to understand what each tool can do

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do and which tool to use
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
"""
                        
                        prompt = PromptTemplate(
                            input_variables=["tools", "input", "agent_scratchpad"],
                            template=react_template,
                            partial_variables={"tool_names": tool_names_str}
                        )
                        
                        agent = create_react_agent(llm, tools, prompt)
                        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
                        
                        response = agent_executor.invoke({"input": request.initial_message})
                        response_text = response.get("output", str(response))
                    except Exception as e:
                        self.logger.warning(f"Agent executor failed: {e}, falling back to direct LLM call")
                        response_text = await llm.ainvoke("\n\n".join([f"{m['role']}: {m['content']}" for m in messages]))
                else:
                    # No tools, use direct LLM call
                    response_text = await llm.ainvoke("\n\n".join([f"{m['role']}: {m['content']}" for m in messages]))

                # Add AI response to conversation
                self.dialogue_manager._add_message_to_conversation(
                    conversation_id, "assistant", response_text
                )

                # Determine if conversation needs to continue
                # Check if response contains questions or asking phrases
                response_lower = response_text.lower()
                
                # Check for completion indicators (AI is ready to proceed)
                completion_phrases = [
                    "thank you", "i have all", "i have the", "i've got", "i've collected",
                    "ready to", "proceed", "all set", "complete", "finished", "done",
                    "i understand", "got it", "perfect", "that's all", "no more questions",
                    "sufficient information", "enough information", "all the information"
                ]
                has_completion_phrase = any(phrase in response_lower for phrase in completion_phrases)
                
                # Check for asking indicators (AI needs more info)
                asking_phrases = ["?", "can you", "could you", "please provide", "i need", 
                                 "what", "which", "when", "where", "how", "tell me",
                                 "missing", "need more", "need additional", "require"]
                is_asking = any(phrase in response_lower for phrase in asking_phrases)
                
                # Determine if more info is needed
                # If AI has completion phrase and is not asking, it's complete
                # If AI is asking questions, it needs more info (unless we've hit max turns)
                needs_more_info = is_asking and not has_completion_phrase and self.dialogue_manager.active_conversations[conversation_id]["turn_number"] < profile.max_turns
                is_complete = has_completion_phrase or (not needs_more_info) or self.dialogue_manager.active_conversations[conversation_id]["turn_number"] >= profile.max_turns

                conversation = self.dialogue_manager.get_conversation(conversation_id)

                return DialogueResponse(
                    conversation_id=conversation_id,
                    turn_number=conversation["turn_number"],
                    max_turns=profile.max_turns,
                    response=response_text,
                    needs_more_info=needs_more_info,
                    is_complete=is_complete,
                    profile_id=profile.id,
                    profile_name=profile.name,
                    model_used=model_name,
                    rag_collection_used=rag_used,
                    conversation_history=conversation["messages"],
                    metadata={
                        "temperature": request.temperature if request.temperature is not None else 0.7,
                        "max_tokens": request.max_tokens if request.max_tokens is not None else 8192,
                        "provider": provider.value,
                    },
                )
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error starting dialogue {dialogue_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/dialogues/{dialogue_id}/continue",
            tags=["Dialogues"],
            summary="Continue Dialogue Conversation",
            description="Continue an existing dialogue conversation by providing a user response. The AI processes the response in context of the full conversation history and either asks for more information or provides a final answer. The turn counter increments with each continue call.",
            response_model=DialogueResponse,
            response_description="AI response with updated conversation status, incremented turn number, and updated conversation history.",
            responses={
                200: {
                    "description": "Dialogue conversation continued successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                                "turn_number": 2,
                                "max_turns": 5,
                                "response": "Thank you! I found your order. What issue are you experiencing with it?",
                                "needs_more_info": True,
                                "is_complete": False,
                                "profile_id": "customer_support_dialogue",
                                "profile_name": "Customer Support Dialogue",
                                "model_used": "gemini-2.5-flash",
                                "rag_collection_used": "support_knowledge",
                                "conversation_history": [
                                    {"role": "user", "content": "I need help with my order", "timestamp": "2024-01-15T10:30:00"},
                                    {"role": "assistant", "content": "I'd be happy to help! Could you please provide your order number?", "timestamp": "2024-01-15T10:30:05"},
                                    {"role": "user", "content": "ORD-12345", "timestamp": "2024-01-15T10:30:30"},
                                    {"role": "assistant", "content": "Thank you! I found your order. What issue are you experiencing with it?", "timestamp": "2024-01-15T10:30:35"}
                                ],
                                "metadata": {
                                    "temperature": 0.7,
                                    "max_tokens": 8192,
                                    "provider": "gemini"
                                }
                            }
                        }
                    }
                },
                400: {"description": "Maximum number of turns reached or invalid request"},
                404: {"description": "Dialogue profile or conversation not found"},
                500: {"description": "Error continuing dialogue conversation"}
            }
        )
        async def continue_dialogue(dialogue_id: str, request: DialogueContinueRequest) -> DialogueResponse:
            """
            **Continue Dialogue Conversation**
            
            Continues an existing dialogue conversation by processing a user's response.
            
            **Execution Process:**
            1. Validates conversation exists and belongs to the dialogue profile
            2. Checks if maximum turns have been reached
            3. Adds user message to conversation history
            4. Increments turn counter
            5. Builds full conversation context (all previous messages)
            6. Calls LLM with conversation history
            7. Analyzes response to determine if more information is needed
            8. Returns updated conversation state
            
            **Request Parameters:**
            - `dialogue_id`: Dialogue profile identifier
            - `conversation_id`: Conversation ID from previous turn (start or continue)
            - `user_message`: User's response to continue the dialogue
            
            **Response Fields:**
            - `conversation_id`: Same conversation ID (for next continue call)
            - `turn_number`: Incremented turn number
            - `response`: AI's response to the user's message
            - `needs_more_info`: Whether AI needs more information
            - `is_complete`: Whether dialogue is complete
            - `conversation_history`: Updated history with all messages
            
            **Turn Management:**
            - Turn number increments with each continue call
            - Maximum turns enforced: returns error if max_turns reached
            - Conversation ends automatically at max_turns even if more info needed
            
            **Use Cases:**
            - Continue customer support conversations
            - Progress through form-filling dialogues
            - Continue consultation sessions
            - Maintain multi-turn AI interactions
            
            **Best Practices:**
            - Always use the conversation_id from the previous response
            - Check `is_complete` before making another continue call
            - Monitor `turn_number` to avoid hitting max_turns
            - Use `needs_more_info` to determine if conversation should continue
            """
            try:
                profile = self.dialogue_manager.get_profile(dialogue_id)
                if not profile:
                    raise HTTPException(status_code=404, detail="Dialogue not found")

                conversation = self.dialogue_manager.get_conversation(request.conversation_id)
                if not conversation or conversation["profile_id"] != dialogue_id:
                    raise HTTPException(status_code=404, detail="Conversation not found")

                # Check if conversation has reached max turns
                if conversation["turn_number"] >= profile.max_turns:
                    raise HTTPException(status_code=400, detail="Maximum number of turns reached")

                # Determine provider/model (same as start)
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
                elif provider_str == "mistral":
                    provider = LLMProviderType.MISTRAL
                    api_key = settings.mistral_api_key
                    model_name = profile.model_name or settings.mistral_default_model
                else:
                    provider = LLMProviderType.GEMINI
                    api_key = settings.gemini_api_key
                    model_name = profile.model_name or settings.gemini_default_model

                # Create LLM caller
                llm_caller = LLMFactory.create_caller(
                    provider=LLMProvider(provider.value),
                    api_key=api_key,
                    model=model_name,
                    temperature=0.7,
                    max_tokens=8192,
                )

                # Wrap in LangChain-compatible wrapper
                llm = LangChainLLMWrapper(llm_caller=llm_caller)

                # Add user message to conversation
                self.dialogue_manager._add_message_to_conversation(
                    request.conversation_id, "user", request.user_message
                )
                self.dialogue_manager._increment_turn(request.conversation_id)

                # Create tools for this dialogue (same as start)
                tools = []
                
                # Add DB tools
                if profile.db_tools and self.db_tools_manager:
                    for tool_id in profile.db_tools:
                        db_profile = self.db_tools_manager.get_profile(tool_id)
                        if db_profile and db_profile.is_active:
                            from langchain.tools import Tool
                            
                            def create_db_tool_func(t_id: str):
                                def db_tool_func(query_input: str) -> str:
                                    try:
                                        if db_profile.allow_dynamic_sql:
                                            result = self.db_tools_manager.execute_query(t_id, sql_input=query_input, force_refresh=True)
                                        else:
                                            result = self.db_tools_manager.execute_query(t_id, force_refresh=True)
                                        
                                        if result.get("rows"):
                                            rows_str = "\n".join([str(row) for row in result["rows"][:10]])
                                            return f"Query executed successfully. Found {result.get('total_rows', len(result.get('rows', [])))} rows.\nColumns: {', '.join(result.get('columns', []))}\nSample rows:\n{rows_str}"
                                        else:
                                            return f"Query executed successfully but returned no rows."
                                    except Exception as e:
                                        return f"Error executing database query: {str(e)}"
                                return db_tool_func
                            
                            tool = Tool(
                                name=f"DB_{tool_id}",
                                func=create_db_tool_func(tool_id),
                                description=f"{db_profile.description or db_profile.name}. Execute database query: {db_profile.sql_statement[:100]}..."
                            )
                            tools.append(tool)
                
                # Add Request tools
                if profile.request_tools and self.request_tools_manager:
                    for tool_id in profile.request_tools:
                        req_profile = self.request_tools_manager.get_profile(tool_id)
                        if req_profile and req_profile.is_active:
                            from langchain.tools import Tool
                            
                            def create_request_tool_func(r_id: str):
                                def request_tool_func(input_data: str = "") -> str:
                                    try:
                                        result = self.request_tools_manager.execute_request(r_id)
                                        if result.get("success"):
                                            return f"Request executed successfully. Status: {result.get('status_code')}. Response: {str(result.get('response_data', {}))[:500]}"
                                        else:
                                            return f"Request failed: {result.get('error', 'Unknown error')}"
                                    except Exception as e:
                                        return f"Error executing request: {str(e)}"
                                return request_tool_func
                            
                            tool = Tool(
                                name=f"Request_{tool_id}",
                                func=create_request_tool_func(tool_id),
                                description=f"{req_profile.description or req_profile.name}. Execute HTTP request: {req_profile.method} {req_profile.url}"
                            )
                            tools.append(tool)

                # Build messages from conversation history
                messages = [{"role": "system", "content": profile.system_prompt}]
                for msg in conversation["messages"]:
                    messages.append({"role": msg.role, "content": msg.content})
                messages.append({"role": "user", "content": request.user_message})

                # Get AI response - use agent executor if tools are available
                if tools:
                    from langchain.agents import AgentExecutor, create_react_agent
                    from langchain.prompts import PromptTemplate
                    
                    # Build tool names list for the prompt
                    tool_names_str = ", ".join([t.name for t in tools])
                    
                    try:
                        # Build conversation history string
                        conv_history = "\n".join([f"{msg.role}: {msg.content}" for msg in conversation["messages"]])
                        
                        # Escape curly braces in system prompt to prevent LangChain from treating them as template variables
                        system_instruction = profile.system_prompt or "You are a helpful AI assistant."
                        # Protect our template variables with placeholders
                        protected_vars = {
                            "{tools}": "___TEMPLATE_VAR_TOOLS___",
                            "{tool_names}": "___TEMPLATE_VAR_TOOL_NAMES___",
                            "{input}": "___TEMPLATE_VAR_INPUT___",
                            "{agent_scratchpad}": "___TEMPLATE_VAR_SCRATCHPAD___"
                        }
                        # Replace protected variables with placeholders
                        for var, placeholder in protected_vars.items():
                            system_instruction = system_instruction.replace(var, placeholder)
                        # Escape all remaining curly braces
                        system_instruction = system_instruction.replace("{", "{{").replace("}", "}}")
                        # Restore our template variables
                        for var, placeholder in protected_vars.items():
                            system_instruction = system_instruction.replace(placeholder, var)
                        
                        system_instruction += " IMPORTANT: You have access to tools that can help you. ALWAYS use the available tools when needed."
                        
                        # Escape curly braces in conversation history as well
                        conv_history = conv_history.replace("{", "{{").replace("}", "}}")
                        
                        react_template = system_instruction + f"""

Previous conversation:
{conv_history}

You have access to the following tools:

{{tools}}

IMPORTANT INSTRUCTIONS:
- ALWAYS use the available tools when they can help answer the question
- Do NOT say you cannot do something if you have a tool that can do it
- ONLY use tools that are listed above - do NOT try to use tools that are not in the list
- Available tool names: {{tool_names}}
- Read tool descriptions carefully to understand what each tool can do

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do and which tool to use
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
{{agent_scratchpad}}
"""
                        
                        prompt = PromptTemplate(
                            input_variables=["tools", "input", "agent_scratchpad"],
                            template=react_template,
                            partial_variables={"tool_names": tool_names_str}
                        )
                        
                        agent = create_react_agent(llm, tools, prompt)
                        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
                        
                        response = agent_executor.invoke({"input": request.user_message})
                        response_text = response.get("output", str(response))
                    except Exception as e:
                        self.logger.warning(f"Agent executor failed: {e}, falling back to direct LLM call")
                        response_text = await llm.ainvoke("\n\n".join([f"{m['role']}: {m['content']}" for m in messages]))
                else:
                    # No tools, use direct LLM call
                    response_text = await llm.ainvoke("\n\n".join([f"{m['role']}: {m['content']}" for m in messages]))

                # Add AI response to conversation
                self.dialogue_manager._add_message_to_conversation(
                    request.conversation_id, "assistant", response_text
                )

                # Get updated conversation
                updated_conversation = self.dialogue_manager.get_conversation(request.conversation_id)

                # Determine if conversation needs to continue
                response_lower = response_text.lower()
                
                # Check for completion indicators (AI is ready to proceed)
                completion_phrases = [
                    "thank you", "i have all", "i have the", "i've got", "i've collected",
                    "ready to", "proceed", "all set", "complete", "finished", "done",
                    "i understand", "got it", "perfect", "that's all", "no more questions",
                    "sufficient information", "enough information", "all the information"
                ]
                has_completion_phrase = any(phrase in response_lower for phrase in completion_phrases)
                
                # Check for asking indicators (AI needs more info)
                asking_phrases = ["?", "can you", "could you", "please provide", "i need", 
                                 "what", "which", "when", "where", "how", "tell me",
                                 "missing", "need more", "need additional", "require"]
                is_asking = any(phrase in response_lower for phrase in asking_phrases)
                
                # Determine if more info is needed
                # If AI has completion phrase and is not asking, it's complete
                # If AI is asking questions, it needs more info (unless we've hit max turns)
                needs_more_info = is_asking and not has_completion_phrase and updated_conversation["turn_number"] < profile.max_turns
                is_complete = has_completion_phrase or (not needs_more_info) or updated_conversation["turn_number"] >= profile.max_turns

                return DialogueResponse(
                    conversation_id=request.conversation_id,
                    turn_number=updated_conversation["turn_number"],
                    max_turns=profile.max_turns,
                    response=response_text,
                    needs_more_info=needs_more_info,
                    is_complete=is_complete,
                    profile_id=profile.id,
                    profile_name=profile.name,
                    model_used=model_name,
                    rag_collection_used=profile.rag_collection,
                    conversation_history=updated_conversation["messages"],
                    metadata={
                        "temperature": 0.7,
                        "max_tokens": 8192,
                        "provider": provider.value,
                    },
                )
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error continuing dialogue {dialogue_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Conversation Endpoints (Multi-AI Conversation Module)
        @self.app.post(
            "/conversations",
            tags=["Conversations"],
            summary="Create Conversation Configuration",
            description="""
            **Create a new conversation configuration profile**
            
            Creates a configuration for multi-AI conversations where two AI models can converse with each other and the user.
            
            **Features:**
            - Two AI models (can be different providers/models)
            - RAG collection support for context
            - Database and Request tool integration
            - Customizable system prompt
            - Configurable maximum turns (5-100)
            
            **Request Parameters:**
            - `name`: Name for the conversation configuration
            - `description`: Optional description
            - `config`: Conversation configuration including model1_config, model2_config (provider, model_name, system_prompt, optional rag_collection), and max_turns (5-100).
            """,
            response_model=Dict[str, Any],
            response_description="Returns id (config_id) and success message.",
        )
        async def create_conversation(req: ConversationCreateRequest):
            """Create a new conversation configuration."""
            try:
                config_id = self.conversation_manager.create_profile(req)
                return {"id": config_id, "message": "Conversation configuration created successfully"}
            except Exception as e:
                self.logger.error(f"Error creating conversation: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/conversations",
            tags=["Conversations"],
            summary="List All Conversation Configurations",
            description="Retrieve all conversation configuration profiles (name, config_id, config with model1_config, model2_config, max_turns).",
            response_model=List[Dict[str, Any]],
            response_description="Array of conversation configuration objects.",
        )
        async def list_conversations():
            """List all conversation configurations."""
            try:
                return self.conversation_manager.list_profiles()
            except Exception as e:
                self.logger.error(f"Error listing conversations: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/conversations/saved",
            tags=["Conversations"],
            summary="List Saved Conversations",
            description="List all saved conversation history files (filenames and metadata).",
            response_model=List[Dict[str, Any]],
            response_description="Array of saved conversation file entries.",
        )
        async def list_saved_conversations():
            """List all saved conversation files."""
            try:
                return self.conversation_manager.list_saved_conversations()
            except Exception as e:
                self.logger.error(f"Error listing saved conversations: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/conversations/saved/{filename:path}",
            tags=["Conversations"],
            summary="Get Saved Conversation Content",
            description="Get the content of a saved conversation file by filename (from list saved).",
            response_model=Dict[str, Any],
            response_description="Object with filename and content (text).",
        )
        async def get_saved_conversation_content(filename: str):
            """Get the content of a saved conversation file."""
            try:
                content = self.conversation_manager.get_saved_conversation_content(filename)
                if content is None:
                    raise HTTPException(status_code=404, detail="Conversation file not found")
                return {"filename": filename, "content": content}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting saved conversation content: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/conversations/{config_id}",
            tags=["Conversations"],
            summary="Get Conversation Configuration",
            description="Retrieve a specific conversation configuration by config_id (full config with model1_config, model2_config, max_turns).",
            response_model=Dict[str, Any],
            response_description="Conversation configuration object.",
        )
        async def get_conversation(config_id: str):
            """Get a conversation configuration."""
            try:
                config = self.conversation_manager.get_profile(config_id)
                if not config:
                    raise HTTPException(status_code=404, detail="Conversation configuration not found")
                return config
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting conversation {config_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put(
            "/conversations/{config_id}",
            tags=["Conversations"],
            summary="Update Conversation Configuration",
            description="Update an existing conversation configuration (name, description, config).",
            response_model=Dict[str, Any],
            response_description="Success message.",
        )
        async def update_conversation(config_id: str, req: ConversationCreateRequest):
            """Update a conversation configuration."""
            try:
                success = self.conversation_manager.update_profile(config_id, req)
                if not success:
                    raise HTTPException(status_code=404, detail="Conversation configuration not found")
                return {"message": "Conversation configuration updated successfully"}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error updating conversation {config_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete(
            "/conversations/{config_id}",
            tags=["Conversations"],
            summary="Delete Conversation Configuration",
            description="Permanently delete a conversation configuration by config_id.",
            response_model=Dict[str, Any],
            response_description="Success message.",
        )
        async def delete_conversation(config_id: str):
            """Delete a conversation configuration."""
            try:
                success = self.conversation_manager.delete_profile(config_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Conversation configuration not found")
                return {"message": "Conversation configuration deleted successfully"}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error deleting conversation {config_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/conversations/start",
            tags=["Conversations"],
            summary="Start Conversation Session",
            description="""
            **Start a new conversation session**
            
            Starts a multi-AI conversation with the specified configuration. The two AI models will alternate turns,
            responding to each other and the user's input.
            
            **Request Parameters:**
            - `config_id`: Conversation configuration ID
            - `topic`: Initial topic or prompt to start the conversation
            
            **Response:**
            - `session_id`: Unique session identifier
            - `turn_number`: Current turn number
            - `max_turns`: Maximum turns allowed
            - `is_complete`: Whether conversation is complete
            - `messages`: Messages from this turn
            - `conversation_history`: Full conversation history
            """,
            response_model=ConversationResponse,
            response_description="Session ID, turn number, messages, and full conversation history.",
        )
        async def start_conversation(req: ConversationStartRequest):
            """Start a new conversation session."""
            try:
                return self.conversation_manager.start_conversation(req)
            except Exception as e:
                import traceback
                self.logger.error(f"Error starting conversation: {e}")
                self.logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/conversations/continue",
            tags=["Conversations"],
            summary="Continue Conversation Session",
            description="""
            **Continue an existing conversation session**
            
            Continues a conversation session. The AI models will continue their dialogue,
            optionally with a user message injected.
            
            **Request Parameters:**
            - `session_id`: Session ID from conversation start
            - `user_message`: Optional user message to inject
            
            **Response:**
            - Updated conversation state with new messages
            """,
            response_model=ConversationResponse,
            response_description="Updated turn number, new messages, and full conversation history.",
        )
        async def continue_conversation(req: ConversationTurnRequest):
            """Continue a conversation session."""
            try:
                return self.conversation_manager.continue_conversation(req)
            except Exception as e:
                self.logger.error(f"Error continuing conversation: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/conversations/history/{session_id}",
            tags=["Conversations"],
            summary="Get Conversation History",
            description="Retrieve the full conversation history for a session (session_id, config_name, started_at, conversation_history).",
            response_model=ConversationHistoryResponse,
            response_description="Session metadata and full list of messages.",
        )
        async def get_conversation_history(session_id: str):
            """Get conversation history for a session."""
            try:
                history = self.conversation_manager.get_conversation_history(session_id)
                if not history:
                    raise HTTPException(status_code=404, detail="Session not found")
                return history
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting conversation history: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Flow Endpoints
        from .models import (
            FlowCreateRequest,
            FlowUpdateRequest,
            FlowExecuteRequest,
            FlowExecuteResponse,
        )

        @self.app.get(
            "/flows",
            tags=["Flows"],
            summary="List All Flows",
            description="Get a list of all workflow flows.",
            response_description="List of flow profiles.",
        )
        async def list_flows():
            """List all workflow flows."""
            try:
                flows = self.flow_service.list_flows()
                return flows
            except Exception as e:
                self.logger.error(f"Error listing flows: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/flows/{flow_id}",
            tags=["Flows"],
            summary="Get Flow",
            description="Get detailed information about a specific flow.",
            response_description="Flow profile details.",
        )
        async def get_flow(flow_id: str):
            """Get a specific flow by ID."""
            try:
                flow = self.flow_service.get_flow(flow_id)
                if not flow:
                    raise HTTPException(status_code=404, detail="Flow not found")
                return flow
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting flow {flow_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/flows",
            tags=["Flows"],
            summary="Create Flow",
            description="Create a new workflow flow that chains together Customization, Agents, DBTools, Requests, and Crawler components.",
            response_description="Flow creation response with flow ID.",
        )
        async def create_flow(req: FlowCreateRequest):
            """Create a new workflow flow."""
            try:
                flow_id = self.flow_service.create_flow(req)
                return {"flow_id": flow_id, "message": "Flow created successfully"}
            except Exception as e:
                self.logger.error(f"Error creating flow: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put(
            "/flows/{flow_id}",
            tags=["Flows"],
            summary="Update Flow",
            description="Update an existing workflow flow.",
            response_description="Confirmation message.",
        )
        async def update_flow(flow_id: str, req: FlowUpdateRequest):
            """Update an existing flow."""
            try:
                success = self.flow_service.update_flow(flow_id, req)
                if not success:
                    raise HTTPException(status_code=404, detail="Flow not found")
                return {"message": "Flow updated successfully"}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error updating flow {flow_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete(
            "/flows/{flow_id}",
            tags=["Flows"],
            summary="Delete Flow",
            description="Delete a workflow flow.",
            response_description="Confirmation message.",
        )
        async def delete_flow(flow_id: str):
            """Delete a flow."""
            try:
                success = self.flow_service.delete_flow(flow_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Flow not found")
                return {"message": "Flow deleted successfully"}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error deleting flow {flow_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/flows/{flow_id}/execute",
            tags=["Flows"],
            summary="Execute Flow",
            description="Execute a workflow flow. Each step uses the output from the previous step as input.",
            response_model=FlowExecuteResponse,
            response_description="Flow execution results including all step outputs.",
        )
        async def execute_flow(flow_id: str, request: FlowExecuteRequest):
            """Execute a workflow flow."""
            try:
                result = await self.flow_service.execute_flow(flow_id, request)
                return result
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                self.logger.error(f"Error executing flow {flow_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        # Dialogue-Driven Flow Endpoints
        @self.app.get(
            "/special-flows-1",
            tags=["Dialogue-Driven Flows"],
            summary="List All Dialogue-Driven Flows",
            description="Get a list of all Dialogue-Driven Flows. Returns all configured Dialogue-Driven Flow profiles with their configurations.",
            response_model=List[SpecialFlow1Profile],
            response_description="List of Dialogue-Driven Flow profiles.",
            responses={
                200: {
                    "description": "Successfully retrieved list of Dialogue-Driven Flows",
                    "content": {
                        "application/json": {
                            "example": [
                                {
                                    "id": "my_special_flow",
                                    "name": "My Special Flow",
                                    "description": "A special flow for data processing",
                                    "config": {
                                        "initial_data_source": {"type": "db_tool", "resource_id": "db1"},
                                        "dialogue_config": {"system_prompt": "You are a helpful assistant", "max_turns_phase1": 5},
                                        "data_fetch_trigger": {"type": "turn_count", "value": 3},
                                        "mid_dialogue_request": {"request_tool_id": "req1"},
                                        "dialogue_phase2": {"continue_same_conversation": True, "max_turns_phase2": 5},
                                        "final_processing": {"system_prompt": "Process the data", "input_template": "{{initial_data}}"},
                                        "final_api_call": {"request_tool_id": "req2", "body_mapping": "{{final_outcome}}"}
                                    },
                                    "is_active": True,
                                    "created_at": "2024-01-01T00:00:00",
                                    "updated_at": "2024-01-01T00:00:00"
                                }
                            ]
                        }
                    }
                },
                500: {"description": "Internal server error"}
            }
        )
        async def list_special_flows_1() -> List[SpecialFlow1Profile]:
            """
            **List All Dialogue-Driven Flows**
            
            Retrieves all Dialogue-Driven Flows configured in the system.
            Returns a list of flow profiles with their complete configurations.
            """
            try:
                flows = self.special_flow_1_service.list_flows()
                return flows
            except Exception as e:
                self.logger.error(f"Error listing dialogue-driven flows: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/special-flows-1/{flow_id}",
            tags=["Dialogue-Driven Flows"],
            summary="Get Dialogue-Driven Flow",
            description="Get detailed information about a specific Dialogue-Driven Flow by its ID. Returns the complete flow profile including all configuration details.",
            response_model=SpecialFlow1Profile,
            response_description="Dialogue-Driven Flow profile details.",
            responses={
                200: {
                    "description": "Successfully retrieved Dialogue-Driven Flow",
                    "content": {
                        "application/json": {
                            "example": {
                                "id": "my_special_flow",
                                "name": "My Special Flow",
                                "description": "A special flow for data processing",
                                "config": {
                                    "initial_data_source": {"type": "db_tool", "resource_id": "db1", "sql_input": None},
                                    "dialogue_config": {
                                        "system_prompt": "You are a helpful assistant",
                                        "max_turns_phase1": 5,
                                        "use_initial_data": True,
                                        "llm_provider": None,
                                        "model_name": None
                                    },
                                    "data_fetch_trigger": {"type": "turn_count", "value": 3},
                                    "mid_dialogue_request": {"request_tool_id": "req1", "param_mapping": {}},
                                    "dialogue_phase2": {
                                        "continue_same_conversation": True,
                                        "inject_fetched_data": True,
                                        "max_turns_phase2": 5
                                    },
                                    "final_processing": {
                                        "system_prompt": "Process the data",
                                        "input_template": "{{initial_data}}\n\nDialogue Summary:\n{{dialogue_summary}}\n\nFetched Data:\n{{fetched_data}}",
                                        "llm_provider": None,
                                        "model_name": None
                                    },
                                    "final_api_call": {"request_tool_id": "req2", "body_mapping": "{{final_outcome}}"}
                                },
                                "is_active": True,
                                "created_at": "2024-01-01T00:00:00",
                                "updated_at": "2024-01-01T00:00:00",
                                "metadata": {}
                            }
                        }
                    }
                },
                404: {"description": "Special Flow 1 not found"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_special_flow_1(flow_id: str) -> SpecialFlow1Profile:
            """
            **Get Dialogue-Driven Flow by ID**
            
            Retrieves detailed information about a specific Dialogue-Driven Flow.
            Includes the complete configuration for all phases of the flow.
            """
            try:
                flow = self.special_flow_1_service.get_flow(flow_id)
                if not flow:
                    raise HTTPException(status_code=404, detail="Special Flow 1 not found")
                return flow
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting dialogue-driven flow {flow_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/special-flows-1",
            tags=["Dialogue-Driven Flows"],
            summary="Create Dialogue-Driven Flow",
            description="Create a new Dialogue-Driven Flow that combines data fetching, dialogue, and API calls. The flow will execute: initial data fetch → dialogue (caches conversation) → fetch data after dialogue → final processing → final API call.",
            response_description="Dialogue-Driven Flow creation response with flow ID.",
            status_code=201,
            responses={
                201: {
                    "description": "Dialogue-Driven Flow created successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "flow_id": "my_special_flow",
                                "message": "Special Flow 1 created successfully"
                            }
                        }
                    }
                },
                400: {"description": "Invalid request data"},
                500: {"description": "Internal server error"}
            }
        )
        async def create_special_flow_1(req: SpecialFlow1CreateRequest):
            """
            **Create a New Dialogue-Driven Flow**
            
            Creates a new Dialogue-Driven Flow with the specified configuration.
            
            **Flow Execution Steps:**
            1. **Initial Data Fetch**: Fetches data from DB tool or Request tool
            2. **Dialogue**: Starts dialogue with initial data (caches all conversation)
            3. **After Dialogue Data Fetch**: Fetches data after dialogue using cached conversation
            4. **Final Processing**: Processes all data using LLM with system prompt (uses cached conversation)
            5. **Final API Call**: Calls final API with processed outcome
            
            **Required Configuration:**
            - Initial data source (DB tool or Request tool)
            - Dialogue system prompt
            - After dialogue request tool
            - Final processing system prompt
            - Final API call request tool
            """
            try:
                flow_id = self.special_flow_1_service.create_flow(req)
                return {"flow_id": flow_id, "message": "Dialogue-Driven Flow created successfully"}
            except Exception as e:
                self.logger.error(f"Error creating dialogue-driven flow: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put(
            "/special-flows-1/{flow_id}",
            tags=["Dialogue-Driven Flows"],
            summary="Update Dialogue-Driven Flow",
            description="Update an existing Dialogue-Driven Flow. All configuration fields can be updated. The flow ID cannot be changed.",
            response_description="Confirmation message.",
            responses={
                200: {
                    "description": "Dialogue-Driven Flow updated successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "message": "Dialogue-Driven Flow updated successfully"
                            }
                        }
                    }
                },
                404: {"description": "Dialogue-Driven Flow not found"},
                400: {"description": "Invalid request data"},
                500: {"description": "Internal server error"}
            }
        )
        async def update_special_flow_1(flow_id: str, req: SpecialFlow1UpdateRequest):
            """
            **Update Dialogue-Driven Flow**
            
            Updates an existing Dialogue-Driven Flow with new configuration.
            All fields in the configuration can be modified.
            """
            try:
                success = self.special_flow_1_service.update_flow(flow_id, req)
                if not success:
                    raise HTTPException(status_code=404, detail="Dialogue-Driven Flow not found")
                return {"message": "Dialogue-Driven Flow updated successfully"}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error updating dialogue-driven flow {flow_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete(
            "/special-flows-1/{flow_id}",
            tags=["Dialogue-Driven Flows"],
            summary="Delete Dialogue-Driven Flow",
            description="Delete a Dialogue-Driven Flow permanently. This action cannot be undone.",
            response_description="Confirmation message.",
            responses={
                200: {
                    "description": "Dialogue-Driven Flow deleted successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "message": "Dialogue-Driven Flow deleted successfully"
                            }
                        }
                    }
                },
                404: {"description": "Dialogue-Driven Flow not found"},
                500: {"description": "Internal server error"}
            }
        )
        async def delete_special_flow_1(flow_id: str):
            """
            **Delete Dialogue-Driven Flow**
            
            Permanently deletes a Dialogue-Driven Flow from the system.
            This action cannot be undone.
            """
            try:
                success = self.special_flow_1_service.delete_flow(flow_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Dialogue-Driven Flow not found")
                return {"message": "Dialogue-Driven Flow deleted successfully"}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error deleting dialogue-driven flow {flow_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/special-flows-1/{flow_id}/execute",
            tags=["Dialogue-Driven Flows"],
            summary="Execute Dialogue-Driven Flow",
            description="Execute a Dialogue-Driven Flow. This will fetch initial data, start dialogue (caches conversation), fetch data after dialogue, process final outcome, and call final API. To resume after dialogue, set resume_from_phase='dialogue' and provide dialogue_phase1_result and initial_data from the previous execution.",
            response_model=SpecialFlow1ExecuteResponse,
            response_description="Dialogue-Driven Flow execution results.",
        )
        async def execute_special_flow_1(flow_id: str, request: SpecialFlow1ExecuteRequest):
            """
            **Execute Dialogue-Driven Flow**
            
            Executes a Dialogue-Driven Flow. The flow will:
            1. Fetch initial data (DB tool or Request tool)
            2. Start dialogue with initial data (caches all conversation)
            3. Pause and wait for user input (if needed)
            4. Fetch data after dialogue using cached conversation
            5. Process final outcome using LLM (uses cached conversation)
            6. Call final API with processed outcome
            
            **Resuming After Dialogue:**
            
            When the flow pauses after dialogue (returns with `phase="dialogue"` and `needs_user_input=True`):
            
            1. Continue the dialogue conversation using `/dialogues/{dialogue_id}/continue` with the `conversation_id` from the response
            2. Once the dialogue is complete (`is_complete=True`), resume the flow by calling this endpoint again with:
               - `resume_from_phase`: "dialogue"
               - `dialogue_phase1_result`: The complete dialogue result from the dialogue API
               - `initial_data`: The `initial_data` from the previous execution response
            
            **Example Resume Request:**
            ```json
            {
              "resume_from_phase": "dialogue",
              "dialogue_phase1_result": {
                "conversation_id": "...",
                "is_complete": true,
                "response": "...",
                "conversation_history": [...]
              },
              "initial_data": {...}
            }
            ```
            """
            try:
                result = await self.special_flow_1_service.execute_flow(flow_id, request)
                return result
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                self.logger.error(f"Error executing dialogue-driven flow {flow_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_image_generation_routes(self):
        """Setup image generation routes"""
        import requests
        from urllib.parse import quote_plus
        from datetime import datetime
        import json
        import uuid
        
        # Create images directory if it doesn't exist
        images_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'generated_images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Images metadata file
        images_meta_file = os.path.join(images_dir, 'images_metadata.json')
        
        def load_images_metadata():
            if os.path.exists(images_meta_file):
                with open(images_meta_file, 'r') as f:
                    return json.load(f)
            return []
        
        def save_images_metadata(metadata):
            with open(images_meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        @self.app.post(
            "/images/generate",
            tags=["Image Generation"],
            summary="Generate Image",
            description="Generate an image from a text prompt using Pollinations API",
        )
        async def generate_image(request: dict):
            """Generate an image from text prompt"""
            try:
                prompt = request.get('prompt', '')
                save_image = request.get('save', True)
                
                if not prompt:
                    raise HTTPException(status_code=400, detail="Prompt is required")
                
                # URL encode the prompt
                encoded_prompt = quote_plus(prompt)
                
                # Build the Pollinations API URL
                image_url = f"https://gen.pollinations.ai/image/{encoded_prompt}?model=flux"
                
                result = {
                    "image_url": image_url,
                    "prompt": prompt,
                }
                
                # Save the image if requested
                if save_image:
                    try:
                        # Generate unique filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        unique_id = str(uuid.uuid4())[:8]
                        filename = f"img_{timestamp}_{unique_id}.png"
                        filepath = os.path.join(images_dir, filename)
                        
                        # Download and save the image
                        headers = {
                            'Authorization': 'Bearer pk_gzM6GGeMB1HQWKgW',
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        }
                        response = requests.get(image_url, headers=headers, timeout=60)
                        
                        if response.status_code == 200:
                            with open(filepath, 'wb') as f:
                                f.write(response.content)
                            
                            # Update metadata
                            metadata = load_images_metadata()
                            metadata.insert(0, {
                                "filename": filename,
                                "prompt": prompt,
                                "created_at": datetime.now().isoformat(),
                                "url": image_url,
                            })
                            save_images_metadata(metadata)
                            
                            result["saved"] = True
                            result["filename"] = filename
                        else:
                            result["saved"] = False
                            result["save_error"] = f"Failed to download image: {response.status_code}"
                    except Exception as e:
                        result["saved"] = False
                        result["save_error"] = str(e)
                
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error generating image: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post(
            "/images/polish-prompt",
            tags=["Image Generation"],
            summary="Polish Image Prompt",
            description="Use AI to enhance and polish an image generation prompt",
        )
        async def polish_image_prompt(request: dict):
            """Polish/enhance an image prompt using AI"""
            try:
                prompt = request.get('prompt', '')
                provider_str = request.get('provider', 'qwen').lower().strip()
                model = request.get('model', '')
                
                if not prompt:
                    raise HTTPException(status_code=400, detail="Prompt is required")
                
                # Get LLM caller with proper provider, api_key, and model
                if provider_str == "gemini":
                    provider = LLMProvider.GEMINI
                    api_key = settings.gemini_api_key
                    if not model:
                        model = settings.gemini_default_model
                elif provider_str == "qwen":
                    provider = LLMProvider.QWEN
                    api_key = settings.qwen_api_key
                    if not model:
                        model = settings.qwen_default_model
                elif provider_str == "mistral":
                    provider = LLMProvider.MISTRAL
                    api_key = settings.mistral_api_key
                    if not model:
                        model = settings.mistral_default_model
                else:
                    # Default to Qwen
                    provider = LLMProvider.QWEN
                    api_key = settings.qwen_api_key
                    model = model or settings.qwen_default_model
                
                llm_caller = LLMFactory.create_caller(provider=provider, api_key=api_key, model=model)
                
                polish_prompt = f"""You are an expert at creating detailed image generation prompts. 
Take the user's basic idea and transform it into a detailed, vivid prompt optimized for AI image generation.

Add details about:
- Subject specifics (appearance, pose, expression)
- Setting/environment (location, time of day, weather)
- Lighting (type, direction, mood)
- Art style (photorealistic, digital art, oil painting, anime, etc.)
- Composition (angle, framing, perspective)
- Mood/atmosphere
- Colors and textures

User's idea: {prompt}

Respond with ONLY the enhanced prompt, nothing else. Make it detailed but concise (2-4 sentences max)."""

                polished = llm_caller.generate(polish_prompt)
                
                # Clean up the response
                polished = polished.strip()
                if polished.startswith('"') and polished.endswith('"'):
                    polished = polished[1:-1]
                
                return {
                    "original_prompt": prompt,
                    "polished_prompt": polished,
                    "provider": provider,
                    "model": model,
                }
                
            except Exception as e:
                self.logger.error(f"Error polishing prompt: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get(
            "/images",
            tags=["Image Generation"],
            summary="Get Generated Images",
            description="Get list of all saved generated images",
        )
        async def get_generated_images():
            """Get all saved generated images"""
            try:
                metadata = load_images_metadata()
                
                # Add full URLs for each image
                for img in metadata:
                    img['url'] = f"/images/file/{img['filename']}"
                
                return metadata
                
            except Exception as e:
                self.logger.error(f"Error getting images: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get(
            "/images/file/{filename}",
            tags=["Image Generation"],
            summary="Get Image File",
            description="Get a specific generated image file",
        )
        async def get_image_file(filename: str):
            """Get a specific image file"""
            from fastapi.responses import FileResponse
            
            try:
                filepath = os.path.join(images_dir, filename)
                
                if not os.path.exists(filepath):
                    raise HTTPException(status_code=404, detail="Image not found")
                
                return FileResponse(filepath, media_type="image/png")
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting image file: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete(
            "/images/{filename}",
            tags=["Image Generation"],
            summary="Delete Generated Image",
            description="Delete a saved generated image",
        )
        async def delete_generated_image(filename: str):
            """Delete a generated image"""
            try:
                filepath = os.path.join(images_dir, filename)
                
                # Delete file if exists
                if os.path.exists(filepath):
                    os.remove(filepath)
                
                # Update metadata
                metadata = load_images_metadata()
                metadata = [img for img in metadata if img['filename'] != filename]
                save_images_metadata(metadata)
                
                return {"message": "Image deleted successfully"}
                
            except Exception as e:
                self.logger.error(f"Error deleting image: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_browser_automation_routes(self):
        """Setup browser automation routes"""
        from .browser_automation import BrowserAutomationTool
        from .models import LLMProviderType
        
        @self.app.post(
            "/browser-automation/execute",
            tags=["Browser Automation"],
            summary="Execute Browser Automation",
            description="Execute browser automation tasks using AI agent with Playwright",
        )
        async def execute_browser_automation(request: dict):
            """Execute browser automation with natural language instructions"""
            try:
                instructions = request.get('instructions', '')
                provider_str = request.get('provider', 'qwen').lower().strip()
                model = request.get('model', '')
                max_steps = request.get('max_steps', 20)
                headless = request.get('headless', False)  # Default to visible browser
                browser_bridge_url = (request.get('browser_bridge_url') or '').strip() or 'ws://localhost:8765'  # default: local browser bridge
                
                if not instructions:
                    raise HTTPException(status_code=400, detail="Instructions are required")
                
                if not model:
                    raise HTTPException(status_code=400, detail="Model is required")
                
                # Map provider string to LLMProviderType
                provider_type = None
                if provider_str == "gemini":
                    provider_type = LLMProviderType.GEMINI
                elif provider_str == "qwen":
                    provider_type = LLMProviderType.QWEN
                elif provider_str == "mistral":
                    provider_type = LLMProviderType.MISTRAL
                else:
                    provider_type = LLMProviderType.QWEN
                
                # Create browser automation tool with specified provider and model
                # If browser_bridge_url is set, AI controls YOUR local browser (run browser_bridge.py on your machine)
                browser_tool = BrowserAutomationTool(
                    llm_provider=provider_type,
                    model_name=model,
                    headless=headless,
                    browser_bridge_url=browser_bridge_url,
                )
                
                # On Windows without bridge, Playwright needs ProactorEventLoop (subprocess support).
                # Run in a dedicated thread with that loop so uvicorn/debugger loop doesn't cause NotImplementedError.
                if sys.platform == "win32" and not browser_bridge_url:
                    def _run_in_proactor_thread():
                        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            return loop.run_until_complete(
                                browser_tool._execute_with_cleanup(instructions, max_steps=max_steps)
                            )
                        finally:
                            loop.close()
                    result = await asyncio.get_running_loop().run_in_executor(
                        None, _run_in_proactor_thread
                    )
                else:
                    result = await browser_tool._execute_with_cleanup(instructions, max_steps=max_steps)
                
                return {
                    "result": result,
                    "instructions": instructions,
                    "provider": provider_str,
                    "model": model,
                    "browser_bridge_url": browser_bridge_url,
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error executing browser automation: {e}")
                import traceback
                error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                raise HTTPException(status_code=500, detail=error_detail)

    def _setup_image_reader_routes(self):
        """Setup image reader routes using Qwen Vision OCR model"""
        image_reader = ImageReader()
        
        @self.app.post(
            "/image-reader/read",
            tags=["Image Reader"],
            summary="Read Text from Image",
            description="Extract text content from a single image using Qwen Vision OCR model (qwen-vl-ocr-2025-11-20)",
            response_description="Extracted text and image metadata",
            responses={
                200: {
                    "description": "Image read successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": True,
                                "text": "Extracted text from image",
                                "image_info": {
                                    "width": 1920,
                                    "height": 1080,
                                    "format": "JPEG",
                                    "mode": "RGB"
                                },
                                "model": "qwen-vl-ocr-2025-11-20",
                                "prompt_used": "Please output only the text content from the image...",
                                "timestamp": "2025-01-23T10:30:00"
                            }
                        }
                    }
                },
                400: {"description": "Invalid request (missing image or invalid format)"},
                500: {"description": "Error processing image"}
            }
        )
        async def read_image(
            file: UploadFile = File(..., description="Image file to read (JPEG, PNG, etc.)"),
            prompt: Optional[str] = None,
            min_pixels: int = 32 * 32 * 3,
            max_pixels: int = 32 * 32 * 8192
        ):
            """
            **Read Text from Image**
            
            Extracts text content from an uploaded image using Qwen's Vision OCR model.
            
            **Parameters:**
            - **file**: Image file (JPEG, PNG, GIF, WebP, etc.)
            - **prompt**: Optional custom prompt for extraction (default: OCR extraction prompt)
            - **min_pixels**: Minimum pixel threshold for image scaling (default: 3072)
            - **max_pixels**: Maximum pixel threshold for image scaling (default: 8388608)
            
            **Returns:**
            - Extracted text content
            - Image metadata (dimensions, format)
            - Processing timestamp
            
            **Example:**
            ```python
            import requests
            
            with open('image.jpg', 'rb') as f:
                response = requests.post(
                    'http://localhost:8000/image-reader/read',
                    files={'file': f},
                    data={'prompt': 'Extract all text from this image'}
                )
            ```
            """
            try:
                # Validate file
                if not file.content_type or not file.content_type.startswith('image/'):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid file type: {file.content_type}. Expected image file."
                    )
                
                # Read image data
                image_data = await file.read()
                
                if len(image_data) == 0:
                    raise HTTPException(status_code=400, detail="Empty image file")
                
                # Process image
                result = image_reader.read_image(
                    image_data=image_data,
                    prompt=prompt,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels
                )
                
                if not result.get("success"):
                    raise HTTPException(
                        status_code=500,
                        detail=result.get("error", "Failed to read image")
                    )
                
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error reading image: {e}")
                import traceback
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing image: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                )

        @self.app.post(
            "/image-reader/read-and-process",
            tags=["Image Reader"],
            summary="Read Image and Process with AI",
            description="Extract text from image (OCR) then process with chosen AI model using a system prompt",
            response_description="Extracted text and AI result",
            responses={
                200: {
                    "description": "Image read and AI processing successful",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": True,
                                "extracted_text": "Text from image...",
                                "ai_result": "AI response based on system prompt...",
                                "provider": "qwen",
                                "model": "qwen-plus",
                                "system_prompt": "Summarize the following content..."
                            }
                        }
                    }
                },
                400: {"description": "Invalid request (missing image, system_prompt, or model)"},
                500: {"description": "Error processing image or AI"}
            }
        )
        async def read_image_and_process(
            file: UploadFile = File(..., description="Image file to read (JPEG, PNG, etc.)"),
            system_prompt: str = Form(..., description="System prompt for AI processing of extracted content"),
            provider: str = Form("qwen", description="AI provider: gemini, qwen, mistral"),
            model: str = Form(..., description="AI model name"),
            ocr_prompt: Optional[str] = Form(None, description="Optional custom prompt for OCR extraction")
        ):
            """
            **Read Image and Process with AI**

            1. Extracts text from the image using Qwen Vision OCR.
            2. Sends the extracted text to the chosen AI model with your system prompt.
            3. Returns both the extracted text and the AI result.
            """
            try:
                if not file.content_type or not file.content_type.startswith('image/'):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid file type: {file.content_type}. Expected image file."
                    )
                if not system_prompt or not system_prompt.strip():
                    raise HTTPException(status_code=400, detail="system_prompt is required")
                if not model or not model.strip():
                    raise HTTPException(status_code=400, detail="model is required")

                image_data = await file.read()
                if len(image_data) == 0:
                    raise HTTPException(status_code=400, detail="Empty image file")

                from .models import LLMProviderType
                provider_str = provider.lower().strip()
                if provider_str == "gemini":
                    provider_type = LLMProviderType.GEMINI
                elif provider_str == "qwen":
                    provider_type = LLMProviderType.QWEN
                elif provider_str == "mistral":
                    provider_type = LLMProviderType.MISTRAL
                else:
                    provider_type = LLMProviderType.QWEN

                result = image_reader.read_and_process(
                    image_data=image_data,
                    system_prompt=system_prompt.strip(),
                    llm_provider=provider_type,
                    model_name=model.strip(),
                    ocr_prompt=ocr_prompt.strip() if ocr_prompt else None
                )

                if not result.get("success"):
                    raise HTTPException(
                        status_code=500,
                        detail=result.get("error", "Failed to read image or process with AI")
                    )
                return result
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error in read_image_and_process: {e}")
                import traceback
                raise HTTPException(
                    status_code=500,
                    detail=f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                )

        @self.app.post(
            "/image-reader/read-and-process-multiple",
            tags=["Image Reader"],
            summary="Read Multiple Images and Process with AI",
            description="Extract text from each image (OCR), concatenate, then process once with the chosen AI model.",
            response_description="Combined extracted text and AI result",
            responses={
                200: {"description": "Images read and AI processing successful"},
                400: {"description": "Invalid request (missing images, system_prompt, or model)"},
                500: {"description": "Error processing images or AI"}
            }
        )
        async def read_image_and_process_multiple(
            files: List[UploadFile] = File(..., description="Image files (1–5)"),
            system_prompt: str = Form(..., description="System prompt for AI processing of combined content"),
            provider: str = Form("qwen", description="AI provider: gemini, qwen, mistral"),
            model: str = Form(..., description="AI model name"),
            ocr_prompt: Optional[str] = Form(None, description="Optional custom prompt for OCR")
        ):
            """OCR each image, concatenate text, then process with AI once."""
            try:
                if not files or len(files) > 5:
                    raise HTTPException(status_code=400, detail="Provide 1 to 5 image files.")
                if not system_prompt or not system_prompt.strip():
                    raise HTTPException(status_code=400, detail="system_prompt is required")
                if not model or not model.strip():
                    raise HTTPException(status_code=400, detail="model is required")
                images_data = []
                for f in files:
                    if not f.content_type or not f.content_type.startswith("image/"):
                        raise HTTPException(status_code=400, detail=f"Invalid file type: {f.content_type}. Expected image.")
                    data = await f.read()
                    if len(data) == 0:
                        raise HTTPException(status_code=400, detail="Empty image file.")
                    images_data.append(data)

                from .models import LLMProviderType
                provider_str = provider.lower().strip()
                if provider_str == "gemini":
                    provider_type = LLMProviderType.GEMINI
                elif provider_str == "qwen":
                    provider_type = LLMProviderType.QWEN
                elif provider_str == "mistral":
                    provider_type = LLMProviderType.MISTRAL
                else:
                    provider_type = LLMProviderType.QWEN

                result = image_reader.read_and_process_multi(
                    images_data=images_data,
                    system_prompt=system_prompt.strip(),
                    llm_provider=provider_type,
                    model_name=model.strip(),
                    ocr_prompt=ocr_prompt.strip() if ocr_prompt else None
                )

                if not result.get("success"):
                    raise HTTPException(
                        status_code=500,
                        detail=result.get("error", "Failed to read images or process with AI")
                    )
                return result
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error in read_image_and_process_multiple: {e}")
                import traceback
                raise HTTPException(
                    status_code=500,
                    detail=f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                )

        @self.app.post(
            "/image-reader/read-multiple",
            tags=["Image Reader"],
            summary="Read Text from Multiple Images",
            description="Extract text content from up to 5 images sequentially using Qwen Vision OCR model",
            response_description="Extracted text from each image",
            responses={
                200: {
                    "description": "Images read successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": True,
                                "total_images": 2,
                                "results": [
                                    {
                                        "success": True,
                                        "text": "Text from image 1",
                                        "image_index": 1,
                                        "image_info": {"width": 1920, "height": 1080},
                                        "timestamp": "2025-01-23T10:30:00"
                                    },
                                    {
                                        "success": True,
                                        "text": "Text from image 2",
                                        "image_index": 2,
                                        "image_info": {"width": 1920, "height": 1080},
                                        "timestamp": "2025-01-23T10:30:01"
                                    }
                                ],
                                "timestamp": "2025-01-23T10:30:01"
                            }
                        }
                    }
                },
                400: {"description": "Invalid request (too many images, missing images, or invalid format)"},
                500: {"description": "Error processing images"}
            }
        )
        async def read_multiple_images(
            files: List[UploadFile] = File(..., description="Image files to read (1-5 images, JPEG, PNG, etc.)"),
            prompt: Optional[str] = None,
            min_pixels: int = 32 * 32 * 3,
            max_pixels: int = 32 * 32 * 8192
        ):
            """
            **Read Text from Multiple Images**
            
            Extracts text content from multiple uploaded images (up to 5) sequentially.
            Each image is processed one by one using Qwen's Vision OCR model.
            
            **Parameters:**
            - **files**: List of image files (1-5 images, JPEG, PNG, GIF, WebP, etc.)
            - **prompt**: Optional custom prompt for extraction (applied to all images)
            - **min_pixels**: Minimum pixel threshold for image scaling (default: 3072)
            - **max_pixels**: Maximum pixel threshold for image scaling (default: 8388608)
            
            **Returns:**
            - Results array with extracted text for each image
            - Image metadata for each image
            - Processing timestamps
            
            **Limitations:**
            - Maximum 5 images per request
            - Images are processed sequentially (one after another)
            
            **Example:**
            ```python
            import requests
            
            files = [
                ('files', open('image1.jpg', 'rb')),
                ('files', open('image2.jpg', 'rb')),
                ('files', open('image3.jpg', 'rb'))
            ]
            
            response = requests.post(
                'http://localhost:8000/image-reader/read-multiple',
                files=files,
                data={'prompt': 'Extract all text from these images'}
            )
            ```
            """
            try:
                # Validate number of files
                if len(files) == 0:
                    raise HTTPException(status_code=400, detail="At least one image file is required")
                
                if len(files) > 5:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Too many images: {len(files)}. Maximum 5 images allowed."
                    )
                
                # Validate all files are images
                images_data = []
                for idx, file in enumerate(files):
                    if not file.content_type or not file.content_type.startswith('image/'):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid file type for image {idx + 1}: {file.content_type}. Expected image file."
                        )
                    
                    image_data = await file.read()
                    if len(image_data) == 0:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Empty image file: {file.filename}"
                        )
                    
                    images_data.append(image_data)
                
                # Process all images
                result = image_reader.read_multiple_images(
                    images_data=images_data,
                    prompt=prompt,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels
                )
                
                if not result.get("success"):
                    raise HTTPException(
                        status_code=500,
                        detail=result.get("error", "Failed to read images")
                    )
                
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error reading multiple images: {e}")
                import traceback
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing images: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                )

    def _setup_pdf_reader_routes(self):
        """Setup PDF reader routes"""
        pdf_reader = PDFReader()
        
        @self.app.post(
            "/pdf-reader/read",
            tags=["PDF Reader"],
            summary="Read and Process PDF",
            description="Extract text from PDF and process it with AI based on system prompt",
            response_description="Extracted text and AI-processed result",
            responses={
                200: {
                    "description": "PDF read and processed successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": True,
                                "extracted_text": "Full text from PDF...",
                                "extracted_text_length": 5000,
                                "page_count": 10,
                                "ai_result": "AI-processed result based on system prompt",
                                "provider": "gemini",
                                "model": "gemini-2.5-flash",
                                "system_prompt": "Summarize the key points",
                                "timestamp": "2025-01-23T10:30:00"
                            }
                        }
                    }
                },
                400: {"description": "Invalid request (missing PDF or invalid format)"},
                500: {"description": "Error processing PDF"}
            }
        )
        async def read_pdf(
            file: UploadFile = File(..., description="PDF file to read"),
            system_prompt: str = Form(..., description="System prompt for AI processing"),
            llm_provider: Optional[str] = Form(None, description="LLM provider (gemini, qwen, mistral)"),
            model_name: Optional[str] = Form(None, description="Model name")
        ):
            """
            **Read and Process PDF**
            
            Extracts text from an uploaded PDF file and processes it with AI based on a system prompt.
            
            **Parameters:**
            - **file**: PDF file to read
            - **system_prompt**: System prompt for AI processing (required)
            - **llm_provider**: Optional LLM provider (gemini, qwen, mistral). Default: system default
            - **model_name**: Optional model name. Default: provider default
            
            **Returns:**
            - Extracted text from PDF
            - AI-processed result based on system prompt
            - PDF metadata (page count, text length)
            - Processing timestamp
            
            **Example:**
            ```python
            import requests
            
            with open('document.pdf', 'rb') as f:
                response = requests.post(
                    'http://localhost:8000/pdf-reader/read',
                    files={'file': f},
                    data={
                        'system_prompt': 'Summarize the key points from this document',
                        'llm_provider': 'gemini',
                        'model_name': 'gemini-2.5-flash'
                    }
                )
            ```
            """
            try:
                # Validate file
                if not file.content_type or file.content_type != 'application/pdf':
                    # Also check filename extension
                    if not file.filename or not file.filename.lower().endswith('.pdf'):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid file type: {file.content_type or 'unknown'}. Expected PDF file."
                        )
                
                # Check if system prompt is provided
                if not system_prompt or not system_prompt.strip():
                    raise HTTPException(
                        status_code=400,
                        detail="System prompt is required"
                    )
                
                # Read PDF data
                pdf_data = await file.read()
                
                if len(pdf_data) == 0:
                    raise HTTPException(status_code=400, detail="Empty PDF file")
                
                # Determine provider
                provider_type = None
                if llm_provider:
                    provider_str = llm_provider.lower().strip()
                    if provider_str == "gemini":
                        provider_type = LLMProviderType.GEMINI
                    elif provider_str == "qwen":
                        provider_type = LLMProviderType.QWEN
                    elif provider_str == "mistral":
                        provider_type = LLMProviderType.MISTRAL
                
                # Process PDF
                result = pdf_reader.read_and_process(
                    pdf_data=pdf_data,
                    system_prompt=system_prompt,
                    llm_provider=provider_type,
                    model_name=model_name
                )
                
                if not result.get("success"):
                    raise HTTPException(
                        status_code=500,
                        detail=result.get("error", "Failed to read PDF")
                    )
                
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error reading PDF: {e}")
                import traceback
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing PDF: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                )

    def _setup_gathering_routes(self):
        """Setup gathering routes: AI-powered data gathering from Wikipedia, Reddit, and web."""
        @self.app.post(
            "/gathering/gather",
            tags=["Gathering"],
            summary="Gather Data",
            description="Use AI to gather information from Wikipedia, Reddit (via web search), and general web search. Has configurable limits to prevent infinite searching.",
            response_model=GatheringResponse,
            response_description="Gathered content in markdown format, or error.",
            responses={
                200: {
                    "description": "Gathering successful",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": True,
                                "content": "## Summary\n\n...",
                                "provider": "qwen",
                                "model": "qwen-plus",
                                "max_iterations": 10,
                            }
                        }
                    }
                },
                400: {"description": "Invalid request (missing prompt)"},
                500: {"description": "Error during gathering"}
            }
        )
        async def gather_data(request: GatheringRequest) -> GatheringResponse:
            """
            **Gather Data from Multiple Sources**

            The AI uses a preset system prompt to gather information in this order:
            1. **Wikipedia** - Factual overview and definitions
            2. **Reddit** - Real discussions (via web search with site:reddit.com)
            3. **Web Search** - Additional sources and news

            **Limits:** max_iterations (default 10) prevents the AI from searching forever.
            """
            try:
                if not request.prompt or not request.prompt.strip():
                    raise HTTPException(status_code=400, detail="prompt is required")

                from .models import LLMProviderType
                provider_type = None
                if request.llm_provider:
                    p = request.llm_provider.lower().strip()
                    if p == "gemini":
                        provider_type = LLMProviderType.GEMINI
                    elif p == "qwen":
                        provider_type = LLMProviderType.QWEN
                    elif p == "mistral":
                        provider_type = LLMProviderType.MISTRAL

                result = await self.gathering_service.gather(
                    prompt=request.prompt.strip(),
                    llm_provider=provider_type,
                    model_name=request.model_name,
                    max_iterations=request.max_iterations or 10,
                    max_tokens=request.max_tokens or 8192,
                    temperature=request.temperature or 0.5,
                )

                return GatheringResponse(
                    success=result.get("success", False),
                    content=result.get("content", ""),
                    provider=result.get("provider"),
                    model=result.get("model"),
                    max_iterations=result.get("max_iterations"),
                    metadata=result.get("metadata", {}),
                    error=result.get("error"),
                )
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error in gathering: {e}")
                import traceback
                raise HTTPException(
                    status_code=500,
                    detail=f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                )

    def _setup_static_handlers(self):
        """Setup handlers for common browser static file requests to prevent 404 errors"""
        
        @self.app.get("/favicon.ico", include_in_schema=False)
        async def favicon():
            """Return empty favicon to prevent 404"""
            return Response(status_code=204)
        
        @self.app.get("/manifest.json", include_in_schema=False)
        async def manifest():
            """Return manifest.json if it exists, otherwise return empty response"""
            manifest_path = os.path.join("frontend", "public", "manifest.json")
            if os.path.exists(manifest_path):
                from fastapi.responses import FileResponse
                return FileResponse(manifest_path, media_type="application/json")
            return Response(status_code=204)
        
        @self.app.get("/logo192.png", include_in_schema=False)
        async def logo192():
            """Return empty logo to prevent 404"""
            return Response(status_code=204)
        
        @self.app.get("/apple-touch-icon.png", include_in_schema=False)
        async def apple_touch_icon():
            """Return empty apple touch icon to prevent 404"""
            return Response(status_code=204)
        
        @self.app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
        async def apple_touch_icon_precomposed():
            """Return empty apple touch icon precomposed to prevent 404"""
            return Response(status_code=204)

    def get_app(self) -> FastAPI:
        """Get the FastAPI application"""
        return self.app


# Create API instance
api = RAGAPI()
app = api.get_app() 