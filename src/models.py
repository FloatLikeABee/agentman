from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import json


class DataFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    TXT = "txt"
    PDF = "pdf"
    DOCX = "docx"


class AgentType(str, Enum):
    RAG = "rag"
    TOOL = "tool"
    HYBRID = "hybrid"


class ToolType(str, Enum):
    EMAIL = "email"
    WEB_SEARCH = "web_search"
    CALCULATOR = "calculator"
    FINANCIAL = "financial"
    WIKIPEDIA = "wikipedia"
    CRAWLER = "crawler"
    EQUALIZER = "equalizer"
    CUSTOM = "custom"


class RAGDataInput(BaseModel):
    name: str = Field(..., description="Name of the RAG data collection")
    description: Optional[str] = Field(None, description="Description of the data")
    format: DataFormat = Field(..., description="Data format")
    content: str = Field(..., description="Data content")
    tags: List[str] = Field(default=[], description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class RAGDataValidation(BaseModel):
    is_valid: bool
    errors: List[str] = Field(default=[])
    warnings: List[str] = Field(default=[])
    record_count: Optional[int] = None


class LLMProviderType(str, Enum):
    GEMINI = "gemini"
    QWEN = "qwen"


class AgentConfig(BaseModel):
    name: str = Field(..., description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    agent_type: AgentType = Field(..., description="Type of agent")
    llm_provider: LLMProviderType = Field(default=LLMProviderType.GEMINI, description="LLM provider to use")
    model_name: str = Field(..., description="Model name (e.g., gemini-2.5-flash, qwen3-max, glm-4.6)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, ge=1, le=32768)
    rag_collections: List[str] = Field(default=[], description="RAG collections to use")
    tools: List[str] = Field(default=[], description="Tools to enable")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    is_active: bool = Field(default=True, description="Whether agent is active")


class ToolConfig(BaseModel):
    name: str = Field(..., description="Tool name")
    tool_type: ToolType = Field(..., description="Type of tool")
    description: str = Field(..., description="Tool description")
    config: Dict[str, Any] = Field(default={}, description="Tool configuration")
    is_active: bool = Field(default=True, description="Whether tool is active")


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    llm_provider: Optional[LLMProviderType] = Field(None, description="Override LLM provider for this query")


class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Query to search in the RAG collection")
    n_results: int = Field(default=5, ge=1, le=100, description="Number of results to return")


class QueryResponse(BaseModel):
    response: str = Field(..., description="Agent response")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source documents")
    metadata: Dict[str, Any] = Field(default={}, description="Response metadata")


class ModelInfo(BaseModel):
    name: str
    size: Optional[Union[str, int]] = None
    modified_at: Optional[str] = None
    digest: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class DirectLLMRequest(BaseModel):
    query: str = Field(..., description="The query to send to the LLM")
    model_name: str = Field(..., description="Model name to use (e.g., gemini-2.5-flash, qwen3-max)")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Temperature for response generation")
    max_tokens: Optional[int] = Field(8192, ge=1, le=32768, description="Maximum tokens in response")
    use_web_search: Optional[bool] = Field(True, description="Whether to enable web search tool")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")


class DirectLLMResponse(BaseModel):
    response: str = Field(..., description="LLM response")
    model_used: str = Field(..., description="Model that was actually used")
    web_search_used: bool = Field(..., description="Whether web search was used")
    metadata: Dict[str, Any] = Field(default={}, description="Response metadata")


class SystemStatus(BaseModel):
    llm_providers_available: List[str]
    default_llm_provider: str
    available_models: List[ModelInfo]
    rag_collections: List[str]
    active_agents: List[str]
    active_tools: List[str]


class CustomizationProfile(BaseModel):
    """Stored customization profile: instructions + optional RAG/LLM config."""

    id: str = Field(..., description="Unique customization id")
    name: str = Field(..., description="Display name")
    description: Optional[str] = Field(None, description="Description / use case")
    system_prompt: str = Field(..., description="Instruction / base command for this customization")
    rag_collection: Optional[str] = Field(
        None,
        description="Optional RAG collection name to use as context",
    )
    llm_provider: Optional[LLMProviderType] = Field(
        None,
        description="Optional LLM provider override for this customization",
    )
    model_name: Optional[str] = Field(
        None,
        description="Optional model override for this customization",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this customization",
    )


class CustomizationCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    system_prompt: str
    rag_collection: Optional[str] = None
    llm_provider: Optional[LLMProviderType] = None
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CustomizationQueryRequest(BaseModel):
    """Query a customization profile with a short user prompt."""

    query: str = Field(..., description="User query to run through the customization")
    n_results: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of RAG documents to pull if rag_collection is set",
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Optional temperature override",
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        le=32768,
        description="Optional max tokens override",
    )


class CustomizationQueryResponse(BaseModel):
    response: str = Field(..., description="LLM response")
    profile_id: str = Field(..., description="Customization profile used")
    profile_name: str = Field(..., description="Customization profile name")
    model_used: str = Field(..., description="Model that was actually used")
    rag_collection_used: Optional[str] = Field(
        None, description="RAG collection used (if any)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")


class DatabaseType(str, Enum):
    SQLSERVER = "sqlserver"
    MYSQL = "mysql"
    MONGODB = "mongodb"


class DatabaseConnectionConfig(BaseModel):
    """Database connection configuration"""
    host: str = Field(..., description="Database host address")
    port: int = Field(..., description="Database port number")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    additional_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional connection parameters (e.g., SSL settings, connection pool settings)"
    )


class DatabaseConnectionConfigUpdate(BaseModel):
    """Database connection configuration for updates (password optional)"""
    host: str = Field(..., description="Database host address")
    port: int = Field(..., description="Database port number")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database username")
    password: Optional[str] = Field(None, description="Database password (optional, omit to keep existing)")
    additional_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional connection parameters (e.g., SSL settings, connection pool settings)"
    )


class DatabaseToolProfile(BaseModel):
    """Stored database tool profile with connection and query configuration"""
    id: str = Field(..., description="Unique database tool id")
    name: str = Field(..., description="Display name")
    description: Optional[str] = Field(None, description="Description / use case")
    db_type: DatabaseType = Field(..., description="Database type (sqlserver, mysql, mongodb)")
    connection_config: DatabaseConnectionConfig = Field(..., description="Database connection configuration")
    sql_statement: str = Field(..., description="SQL query statement (for SQL databases) or query (for MongoDB)")
    is_active: bool = Field(default=True, description="Whether this database tool is active")
    cache_ttl_hours: float = Field(default=1.0, description="Cache TTL in hours (default: 1 hour)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this database tool"
    )


class DatabaseToolCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    db_type: DatabaseType
    connection_config: DatabaseConnectionConfig
    sql_statement: str
    is_active: bool = Field(default=True)
    cache_ttl_hours: float = Field(default=1.0, ge=0.1, le=24.0, description="Cache TTL in hours (0.1 to 24 hours)")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatabaseToolUpdateRequest(BaseModel):
    """Update request for database tools (password optional)"""
    name: str
    description: Optional[str] = None
    db_type: DatabaseType
    connection_config: DatabaseConnectionConfigUpdate
    sql_statement: str
    is_active: bool = Field(default=True)
    cache_ttl_hours: float = Field(default=1.0, ge=0.1, le=24.0, description="Cache TTL in hours (0.1 to 24 hours)")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatabaseToolPreviewResponse(BaseModel):
    """Preview response showing first 10 rows of query results"""
    tool_id: str = Field(..., description="Database tool ID")
    tool_name: str = Field(..., description="Database tool name")
    columns: List[str] = Field(..., description="Column names")
    rows: List[List[Any]] = Field(..., description="First 10 rows of data")
    total_rows: Optional[int] = Field(None, description="Total number of rows (if available)")
    cached: bool = Field(..., description="Whether data is from cache")
    cache_expires_at: Optional[str] = Field(None, description="Cache expiration timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RequestType(str, Enum):
    HTTP = "http"
    INTERNAL = "internal"


class HTTPMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class RequestConfig(BaseModel):
    """Request configuration for HTTP or internal service calls"""
    name: str = Field(..., description="Unique request name/task identifier")
    description: Optional[str] = Field(None, description="Request description")
    request_type: RequestType = Field(..., description="Type of request (http or internal)")
    method: Optional[HTTPMethod] = Field(None, description="HTTP method (required for HTTP requests)")
    url: Optional[str] = Field(None, description="HTTP URL (required for HTTP requests)")
    endpoint: Optional[str] = Field(None, description="Internal service endpoint (required for internal requests)")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    params: Dict[str, Any] = Field(default_factory=dict, description="URL query parameters")
    body: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Request body (string or JSON object)")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RequestProfile(BaseModel):
    """Stored request profile with configuration and last response"""
    id: str = Field(..., description="Unique request ID")
    name: str = Field(..., description="Request name/task identifier")
    description: Optional[str] = Field(None, description="Request description")
    request_type: RequestType = Field(..., description="Type of request")
    method: Optional[HTTPMethod] = Field(None, description="HTTP method")
    url: Optional[str] = Field(None, description="HTTP URL")
    endpoint: Optional[str] = Field(None, description="Internal service endpoint")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    params: Dict[str, Any] = Field(default_factory=dict, description="URL query parameters")
    body: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Request body")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    last_response: Optional[Dict[str, Any]] = Field(None, description="Last response data")
    last_executed_at: Optional[str] = Field(None, description="Last execution timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RequestCreateRequest(BaseModel):
    """Request to create a new request configuration"""
    name: str = Field(..., description="Unique request name/task identifier")
    description: Optional[str] = None
    request_type: RequestType
    method: Optional[HTTPMethod] = None
    url: Optional[str] = None
    endpoint: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    body: Optional[Union[str, Dict[str, Any]]] = None
    timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RequestUpdateRequest(BaseModel):
    """Request to update an existing request configuration"""
    name: str = Field(..., description="Unique request name/task identifier")
    description: Optional[str] = None
    request_type: RequestType
    method: Optional[HTTPMethod] = None
    url: Optional[str] = None
    endpoint: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    body: Optional[Union[str, Dict[str, Any]]] = None
    timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RequestExecuteResponse(BaseModel):
    """Response from executing a request"""
    request_id: str = Field(..., description="Request ID")
    request_name: str = Field(..., description="Request name")
    success: bool = Field(..., description="Whether request was successful")
    status_code: Optional[int] = Field(None, description="HTTP status code")
    response_data: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Response data")
    response_headers: Dict[str, str] = Field(default_factory=dict, description="Response headers")
    execution_time: float = Field(..., description="Execution time in seconds")
    error: Optional[str] = Field(None, description="Error message if request failed")
    executed_at: str = Field(..., description="Execution timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CrawlerRequest(BaseModel):
    url: str = Field(..., description="URL to crawl")
    use_js: bool = Field(default=False, description="Use JavaScript rendering (Playwright/Selenium)")
    llm_provider: Optional[str] = Field(None, description="LLM provider to use for extraction (gemini/qwen)")
    model: Optional[str] = Field(None, description="Model name to use")
    collection_name: Optional[str] = Field(None, description="Override AI-generated collection name")
    collection_description: Optional[str] = Field(None, description="Override AI-generated collection description")


class CrawlerResponse(BaseModel):
    success: bool
    url: str
    collection_name: Optional[str] = None
    collection_description: Optional[str] = None
    raw_file: Optional[str] = None
    extracted_file: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
