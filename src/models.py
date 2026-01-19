from pydantic import BaseModel, Field, field_validator, model_validator
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
    DOCUMENT_READER = "document_reader"
    YOUTUBE_SUMMARIZER = "youtube_summarizer"
    ACADEMIC_SEARCH = "academic_search"
    MIND_MAP = "mind_map"
    DEBATE_ANALYZER = "debate_analyzer"
    FIRST_PRINCIPLES = "first_principles"
    IMAGE_GENERATOR = "image_generator"
    STORY_GENERATOR = "story_generator"
    TASK_PLANNER = "task_planner"
    MULTI_AGENT = "multi_agent"
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
    MISTRAL = "mistral"


class SmartImportRequest(BaseModel):
    """Request for Smart Import functionality"""
    file_content: str = Field(..., description="File content as string (CSV or JSON)")
    file_format: str = Field(..., description="File format: 'csv' or 'json'")
    llm_provider: Optional[LLMProviderType] = Field(None, description="LLM provider to use for processing (default: system default)")
    model_name: Optional[str] = Field(None, description="Model name to use (default: provider default)")
    processing_instructions: Optional[str] = Field(None, description="Optional custom instructions for AI processing")
    auto_name: bool = Field(default=True, description="Automatically generate collection name using AI")


class SmartImportResponse(BaseModel):
    """Response from Smart Import"""
    success: bool = Field(..., description="Whether import was successful")
    collection_name: str = Field(..., description="Name of the created/used RAG collection")
    collection_description: str = Field(..., description="Description of the collection")
    processed_data: Optional[str] = Field(None, description="Processed/cleaned data in RAG format")
    original_record_count: Optional[int] = Field(None, description="Number of records in original file")
    processed_record_count: Optional[int] = Field(None, description="Number of records after processing")
    message: str = Field(..., description="Status message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


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
    system_prompt_data: Optional[str] = Field(None, description="Data to inject into system prompt (replaces {data} placeholder)")
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
    SQLITE = "sqlite"


class DatabaseConnectionConfig(BaseModel):
    """Database connection configuration
    
    For SQLite: 
    - Use 'database' field for the SQLite file path (e.g., '/path/to/database.db' or './data.db')
    - 'host', 'port', 'username', 'password' are not used
    - 'additional_params' can include: timeout, check_same_thread, isolation_level
    """
    host: str = Field(..., description="Database host address (for SQLite: can be used as alternative to 'database' field for file path)")
    port: int = Field(..., description="Database port number (not used for SQLite)")
    database: str = Field(..., description="Database name (for SQLite: database file path, e.g., '/path/to/database.db')")
    username: str = Field(..., description="Database username (not used for SQLite)")
    password: str = Field(..., description="Database password (not used for SQLite)")
    additional_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional connection parameters. For SQLite: timeout (float), check_same_thread (bool), isolation_level (str). For others: SSL settings, connection pool settings."
    )


class DatabaseConnectionConfigUpdate(BaseModel):
    """Database connection configuration for updates (password optional)
    
    For SQLite: 
    - Use 'database' field for the SQLite file path
    - 'host', 'port', 'username', 'password' are not used
    """
    host: str = Field(..., description="Database host address (for SQLite: can be used as alternative to 'database' field for file path)")
    port: int = Field(..., description="Database port number (not used for SQLite)")
    database: str = Field(..., description="Database name (for SQLite: database file path)")
    username: str = Field(..., description="Database username (not used for SQLite)")
    password: Optional[str] = Field(None, description="Database password (optional, omit to keep existing, not used for SQLite)")
    additional_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional connection parameters. For SQLite: timeout, check_same_thread, isolation_level."
    )


class DatabaseToolProfile(BaseModel):
    """Stored database tool profile with connection and query configuration"""
    id: str = Field(..., description="Unique database tool id")
    name: str = Field(..., description="Display name")
    description: Optional[str] = Field(None, description="Description / use case")
    db_type: DatabaseType = Field(..., description="Database type (sqlserver, mysql, sqlite, mongodb)")
    connection_config: DatabaseConnectionConfig = Field(..., description="Database connection configuration")
    sql_statement: str = Field(..., description="SQL query statement (for SQL databases) or query (for MongoDB)")
    is_active: bool = Field(default=True, description="Whether this database tool is active")
    cache_ttl_hours: float = Field(default=1.0, description="Cache TTL in hours (default: 1 hour)")
    allow_dynamic_sql: bool = Field(default=False, description="Allow dynamic SQL input at execution time")
    preset_sql_statement: Optional[str] = Field(None, description="Optional preset/base SQL statement. If provided and allow_dynamic_sql is True, dynamic input will be appended as WHERE condition or used as full SQL if preset is empty.")
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
    allow_dynamic_sql: bool = Field(default=False, description="Allow dynamic SQL input at execution time")
    preset_sql_statement: Optional[str] = Field(None, description="Optional preset/base SQL statement. If provided and allow_dynamic_sql is True, dynamic input will be appended as WHERE condition.")
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
    allow_dynamic_sql: bool = Field(default=False, description="Allow dynamic SQL input at execution time")
    preset_sql_statement: Optional[str] = Field(None, description="Optional preset/base SQL statement. If provided and allow_dynamic_sql is True, dynamic input will be appended as WHERE condition.")
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


class DatabaseToolExecuteRequest(BaseModel):
    """Request model for executing database query with dynamic SQL"""
    sql_input: Optional[str] = Field(None, description="Optional dynamic SQL input (WHERE condition or full SQL statement)")


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
    body: Optional[Union[str, Dict[str, Any], List[Any]]] = Field(None, description="Request body (string, JSON object, or array)")
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
    body: Optional[Union[str, Dict[str, Any], List[Any]]] = Field(None, description="Request body (string, JSON object, or array)")
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
    body: Optional[Union[str, Dict[str, Any], List[Any]]] = None
    timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('description', mode='before')
    @classmethod
    def convert_empty_description(cls, v):
        """Convert empty strings to None for description"""
        if v == "":
            return None
        return v
    
    @field_validator('url', mode='before')
    @classmethod
    def convert_empty_url(cls, v):
        """Convert empty strings to None for url"""
        if v == "":
            return None
        return v
    
    @field_validator('endpoint', mode='before')
    @classmethod
    def convert_empty_endpoint(cls, v):
        """Convert empty strings to None for endpoint"""
        if v == "":
            return None
        return v
    
    @field_validator('body', mode='before')
    @classmethod
    def convert_empty_body(cls, v):
        """Convert empty strings to None for body field, accept lists/arrays"""
        if v == "":
            return None
        # Accept lists, dicts, or strings as-is
        return v
    
    @field_validator('timeout', mode='before')
    @classmethod
    def convert_timeout(cls, v):
        """Convert integer timeout to float if needed"""
        if v is None:
            return 30.0
        if isinstance(v, int):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return 30.0
        return v
    
    @field_validator('request_type', mode='before')
    @classmethod
    def normalize_request_type(cls, v):
        """Normalize request_type to enum value"""
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower == 'http':
                return RequestType.HTTP
            elif v_lower == 'internal':
                return RequestType.INTERNAL
        return v
    
    @field_validator('method', mode='before')
    @classmethod
    def normalize_method(cls, v):
        """Normalize HTTP method to enum value"""
        if v is None:
            return None
        if isinstance(v, str) and v:
            v_upper = v.upper()
            try:
                return HTTPMethod(v_upper)
            except ValueError:
                # Return None if invalid method instead of raising error
                return None
        return v
    
    @model_validator(mode='before')
    @classmethod
    def validate_model(cls, data):
        """Pre-process the entire model data"""
        if isinstance(data, dict):
            # Convert empty strings to None for optional fields
            for field in ['description', 'url', 'endpoint']:
                if field in data and data[field] == "":
                    data[field] = None
            
            # Handle body - convert empty string to None, allow arrays/dicts/strings
            if 'body' in data:
                if data['body'] == "":
                    data['body'] = None
                # Arrays, dicts, and strings are all valid - keep as-is
            
            # Convert timeout to float
            if 'timeout' in data:
                if data['timeout'] is None:
                    data['timeout'] = 30.0
                elif isinstance(data['timeout'], int):
                    data['timeout'] = float(data['timeout'])
                elif isinstance(data['timeout'], str):
                    try:
                        data['timeout'] = float(data['timeout'])
                    except ValueError:
                        data['timeout'] = 30.0
            
            # Normalize request_type
            if 'request_type' in data and isinstance(data['request_type'], str):
                v_lower = data['request_type'].lower()
                if v_lower == 'http':
                    data['request_type'] = RequestType.HTTP
                elif v_lower == 'internal':
                    data['request_type'] = RequestType.INTERNAL
            
            # Normalize method
            if 'method' in data and data['method']:
                if isinstance(data['method'], str):
                    try:
                        data['method'] = HTTPMethod(data['method'].upper())
                    except ValueError:
                        data['method'] = None
        
        return data


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
    body: Optional[Union[str, Dict[str, Any], List[Any]]] = None
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
    follow_links: bool = Field(default=False, description="Follow links recursively to crawl entire site")
    max_depth: int = Field(default=3, ge=1, le=10, description="Maximum depth for recursive crawling (1-10)")
    max_pages: int = Field(default=50, ge=1, le=1000, description="Maximum number of pages to crawl (1-1000)")
    same_domain_only: bool = Field(default=True, description="Only follow links within the same domain")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom HTTP headers (e.g., Authorization tokens)")


class CrawlerProfile(BaseModel):
    """Crawler profile configuration"""
    id: str = Field(..., description="Unique profile ID")
    name: str = Field(..., description="Profile name")
    description: Optional[str] = Field(None, description="Profile description")
    url: str = Field(..., description="URL to crawl")
    use_js: bool = Field(default=False, description="Use JavaScript rendering")
    llm_provider: Optional[str] = Field(None, description="LLM provider for extraction")
    model: Optional[str] = Field(None, description="Model name")
    collection_name: Optional[str] = Field(None, description="RAG collection name")
    collection_description: Optional[str] = Field(None, description="RAG collection description")
    follow_links: bool = Field(default=False, description="Follow links recursively")
    max_depth: int = Field(default=3, ge=1, le=10, description="Maximum crawl depth")
    max_pages: int = Field(default=50, ge=1, le=1000, description="Maximum pages to crawl")
    same_domain_only: bool = Field(default=True, description="Only follow same domain links")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom HTTP headers")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class CrawlerCreateRequest(BaseModel):
    """Request to create a crawler profile"""
    name: str = Field(..., description="Profile name")
    description: Optional[str] = Field(None, description="Profile description")
    url: str = Field(..., description="URL to crawl")
    use_js: bool = Field(default=False, description="Use JavaScript rendering")
    llm_provider: Optional[str] = Field(None, description="LLM provider for extraction")
    model: Optional[str] = Field(None, description="Model name")
    collection_name: Optional[str] = Field(None, description="RAG collection name")
    collection_description: Optional[str] = Field(None, description="RAG collection description")
    follow_links: bool = Field(default=False, description="Follow links recursively")
    max_depth: int = Field(default=3, ge=1, le=10, description="Maximum crawl depth")
    max_pages: int = Field(default=50, ge=1, le=1000, description="Maximum pages to crawl")
    same_domain_only: bool = Field(default=True, description="Only follow same domain links")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom HTTP headers")


class CrawlerUpdateRequest(BaseModel):
    """Request to update a crawler profile"""
    name: Optional[str] = Field(None, description="Profile name")
    description: Optional[str] = Field(None, description="Profile description")
    url: Optional[str] = Field(None, description="URL to crawl")
    use_js: Optional[bool] = Field(None, description="Use JavaScript rendering")
    llm_provider: Optional[str] = Field(None, description="LLM provider for extraction")
    model: Optional[str] = Field(None, description="Model name")
    collection_name: Optional[str] = Field(None, description="RAG collection name")
    collection_description: Optional[str] = Field(None, description="RAG collection description")
    follow_links: Optional[bool] = Field(None, description="Follow links recursively")
    max_depth: Optional[int] = Field(None, ge=1, le=10, description="Maximum crawl depth")
    max_pages: Optional[int] = Field(None, ge=1, le=1000, description="Maximum pages to crawl")
    same_domain_only: Optional[bool] = Field(None, description="Only follow same domain links")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom HTTP headers")


class CrawlerResponse(BaseModel):
    success: bool
    url: str
    collection_name: Optional[str] = None
    collection_description: Optional[str] = None
    raw_file: Optional[str] = None
    extracted_file: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    pages_crawled: Optional[int] = None
    total_links_found: Optional[int] = None


class FlowStepType(str, Enum):
    """Types of steps in a flow"""
    CUSTOMIZATION = "customization"
    AGENT = "agent"
    DB_TOOL = "db_tool"
    REQUEST = "request"
    CRAWLER = "crawler"
    DIALOGUE = "dialogue"


class FlowStepConfig(BaseModel):
    """Configuration for a single step in a flow"""
    step_id: str = Field(..., description="Unique step identifier within the flow")
    step_type: FlowStepType = Field(..., description="Type of step")
    step_name: str = Field(..., description="Display name for this step")
    resource_id: str = Field(..., description="ID of the resource to use (customization_id, agent_id, db_tool_id, request_id)")
    input_query: Optional[str] = Field(None, description="Input query/prompt for this step (if not using previous step output)")
    use_previous_output: bool = Field(default=False, description="Whether to use output from previous step as input")
    output_mapping: Optional[Dict[str, str]] = Field(None, description="Mapping previous step output to this step's parameters")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional step metadata")


class FlowConfig(BaseModel):
    """Configuration for a complete flow"""
    name: str = Field(..., description="Flow name")
    description: Optional[str] = Field(None, description="Flow description")
    steps: List[FlowStepConfig] = Field(..., description="Ordered list of flow steps")
    is_active: bool = Field(default=True, description="Whether flow is active")


class FlowProfile(BaseModel):
    """Stored flow profile"""
    id: str = Field(..., description="Unique flow ID")
    name: str = Field(..., description="Flow name")
    description: Optional[str] = Field(None, description="Flow description")
    steps: List[FlowStepConfig] = Field(..., description="Ordered list of flow steps")
    is_active: bool = Field(default=True, description="Whether flow is active")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FlowCreateRequest(BaseModel):
    """Request to create a new flow"""
    name: str = Field(..., description="Flow name")
    description: Optional[str] = None
    steps: List[FlowStepConfig] = Field(..., description="Ordered list of flow steps")
    is_active: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FlowUpdateRequest(BaseModel):
    """Request to update an existing flow"""
    name: str = Field(..., description="Flow name")
    description: Optional[str] = None
    steps: List[FlowStepConfig] = Field(..., description="Ordered list of flow steps")
    is_active: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FlowExecuteRequest(BaseModel):
    """Request to execute a flow"""
    initial_input: Optional[str] = Field(None, description="Initial input for the first step (if needed)")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for flow execution")
    resume_from_step: Optional[int] = Field(None, description="Step index (1-based) to resume from (for paused flows)")
    previous_step_results: Optional[List["FlowStepResult"]] = Field(None, description="Previous step results when resuming")


class FlowStepResult(BaseModel):
    """Result from executing a single flow step"""
    step_id: str = Field(..., description="Step ID")
    step_name: str = Field(..., description="Step name")
    step_type: FlowStepType = Field(..., description="Step type")
    success: bool = Field(..., description="Whether step executed successfully")
    output: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Step output")
    error: Optional[str] = Field(None, description="Error message if step failed")
    execution_time: float = Field(..., description="Execution time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FlowExecuteResponse(BaseModel):
    """Response from executing a flow"""
    flow_id: str = Field(..., description="Flow ID")
    flow_name: str = Field(..., description="Flow name")
    success: bool = Field(..., description="Whether flow completed successfully")
    step_results: List[FlowStepResult] = Field(..., description="Results from each step")
    final_output: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Final output from last step")
    total_execution_time: float = Field(..., description="Total execution time in seconds")
    error: Optional[str] = Field(None, description="Error message if flow failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# Dialogue Models
class DialogueMessage(BaseModel):
    """A single message in a dialogue conversation"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")


class DialogueProfile(BaseModel):
    """Stored dialogue profile: multi-turn conversation with system prompt"""
    id: str = Field(..., description="Unique dialogue id")
    name: str = Field(..., description="Display name")
    description: Optional[str] = Field(None, description="Description / use case")
    system_prompt: str = Field(..., description="System prompt / instruction for the dialogue")
    rag_collection: Optional[str] = Field(
        None,
        description="Optional RAG collection name to use as context",
    )
    db_tools: List[str] = Field(
        default_factory=list,
        description="List of database tool IDs to use in this dialogue",
    )
    request_tools: List[str] = Field(
        default_factory=list,
        description="List of request tool IDs to use in this dialogue",
    )
    llm_provider: Optional[LLMProviderType] = Field(
        None,
        description="Optional LLM provider override for this dialogue",
    )
    model_name: Optional[str] = Field(
        None,
        description="Optional model override for this dialogue",
    )
    max_turns: int = Field(default=30, ge=1, le=30, description="Maximum number of conversation turns (default: 30)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this dialogue",
    )


class DialogueCreateRequest(BaseModel):
    """Request to create a new dialogue"""
    name: str
    description: Optional[str] = None
    system_prompt: str
    rag_collection: Optional[str] = None
    db_tools: List[str] = Field(default_factory=list, description="List of database tool IDs to use")
    request_tools: List[str] = Field(default_factory=list, description="List of request tool IDs to use")
    llm_provider: Optional[LLMProviderType] = None
    model_name: Optional[str] = None
    max_turns: int = Field(default=30, ge=1, le=30)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DialogueUpdateRequest(BaseModel):
    """Request to update an existing dialogue"""
    name: str
    description: Optional[str] = None
    system_prompt: str
    rag_collection: Optional[str] = None
    db_tools: List[str] = Field(default_factory=list, description="List of database tool IDs to use")
    request_tools: List[str] = Field(default_factory=list, description="List of request tool IDs to use")
    llm_provider: Optional[LLMProviderType] = None
    model_name: Optional[str] = None
    max_turns: int = Field(default=30, ge=1, le=30)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DialogueStartRequest(BaseModel):
    """Request to start a new dialogue conversation"""
    initial_message: str = Field(..., description="Initial user message to start the dialogue")
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


class DialogueContinueRequest(BaseModel):
    """Request to continue an existing dialogue conversation"""
    user_message: str = Field(..., description="User's response to continue the dialogue")
    conversation_id: str = Field(..., description="Conversation ID from previous turn")


class DialogueResponse(BaseModel):
    """Response from a dialogue turn"""
    conversation_id: str = Field(..., description="Unique conversation ID for this dialogue session")
    turn_number: int = Field(..., description="Current turn number (1-based)")
    max_turns: int = Field(..., description="Maximum turns allowed")
    response: str = Field(..., description="AI's response")
    needs_more_info: bool = Field(..., description="Whether AI is asking for more information (conversation continues)")
    is_complete: bool = Field(..., description="Whether the dialogue is complete (final response provided)")
    profile_id: str = Field(..., description="Dialogue profile used")
    profile_name: str = Field(..., description="Dialogue profile name")
    model_used: str = Field(..., description="Model that was actually used")
    rag_collection_used: Optional[str] = Field(
        None, description="RAG collection used (if any)"
    )
    conversation_history: List[DialogueMessage] = Field(..., description="Full conversation history up to this point")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")


# Dialogue-Driven Flow Models
class InitialDataSourceConfig(BaseModel):
    """Configuration for initial data source"""
    type: str = Field(..., description="Type: 'db_tool' or 'request_tool'")
    resource_id: str = Field(..., description="ID of the DB tool or Request tool")
    sql_input: Optional[str] = Field(None, description="Optional SQL input for DB tool")


class DialogueConfig(BaseModel):
    """Configuration for dialogue phase"""
    system_prompt: str = Field(..., description="System prompt for the dialogue")
    max_turns_phase1: int = Field(default=5, ge=1, le=20, description="Maximum turns before data fetch")
    use_initial_data: bool = Field(default=True, description="Inject initial data into dialogue context")
    llm_provider: Optional[LLMProviderType] = Field(None, description="LLM provider override")
    model_name: Optional[str] = Field(None, description="Model name override")


class DataFetchTrigger(BaseModel):
    """Configuration for when to trigger mid-dialogue data fetch"""
    type: str = Field(..., description="Trigger type: 'turn_count', 'keyword', 'user_trigger', 'ai_detected'")
    value: Optional[Union[int, str]] = Field(None, description="Turn count (int) or keyword (str) depending on type")


class MidDialogueRequestConfig(BaseModel):
    """Configuration for mid-dialogue request"""
    request_tool_id: str = Field(..., description="Request tool ID to use")
    param_mapping: Optional[Union[str, Dict[str, str]]] = Field(
        None,
        description="Mapping of request params from dialogue context. Can be a template string like '{{dialogue.response}}' (if response is JSON, it will be parsed and merged), or a dict mapping param keys to template values. Use {{dialogue.user_input}}, {{dialogue.conversation_history}}, etc."
    )


class DialoguePhase2Config(BaseModel):
    """Configuration for dialogue phase 2 (after data fetch)"""
    continue_same_conversation: bool = Field(default=True, description="Continue same conversation or start new")
    inject_fetched_data: bool = Field(default=True, description="Inject fetched data into dialogue context")
    max_turns_phase2: int = Field(default=5, ge=1, le=20, description="Maximum turns in phase 2")


class FinalProcessingConfig(BaseModel):
    """Configuration for final processing step"""
    system_prompt: str = Field(..., description="System prompt for final processing")
    input_template: str = Field(
        default="{{initial_data}}\n\nDialogue Summary:\n{{dialogue_summary}}\n\nFetched Data:\n{{fetched_data}}",
        description="Template for combining all data. Use {{initial_data}}, {{dialogue_summary}}, {{fetched_data}}"
    )
    llm_provider: Optional[LLMProviderType] = Field(None, description="LLM provider override")
    model_name: Optional[str] = Field(None, description="Model name override")


class FinalAPICallConfig(BaseModel):
    """Configuration for final API call"""
    request_tool_id: str = Field(..., description="Request tool ID for final API call")
    body_mapping: str = Field(
        default="{{final_outcome}}",
        description="How to format final outcome for API. Use {{final_outcome}} or JSON template"
    )


class SpecialFlow1Config(BaseModel):
    """Configuration for Dialogue-Driven Flow"""
    initial_data_source: InitialDataSourceConfig = Field(..., description="Initial data source configuration")
    dialogue_config: DialogueConfig = Field(..., description="Dialogue configuration")
    data_fetch_trigger: Optional[DataFetchTrigger] = Field(None, description="When to trigger data fetch (deprecated - not used in simplified flow)")
    mid_dialogue_request: MidDialogueRequestConfig = Field(..., description="Request configuration for fetching data after dialogue")
    dialogue_phase2: Optional[DialoguePhase2Config] = Field(None, description="Dialogue phase 2 configuration (deprecated - not used in simplified flow)")
    final_processing: FinalProcessingConfig = Field(..., description="Final processing configuration")
    final_api_call: FinalAPICallConfig = Field(..., description="Final API call configuration")


class SpecialFlow1Profile(BaseModel):
    """Stored Dialogue-Driven Flow profile"""
    id: str = Field(..., description="Unique flow ID")
    name: str = Field(..., description="Flow name")
    description: Optional[str] = Field(None, description="Flow description")
    config: SpecialFlow1Config = Field(..., description="Flow configuration")
    is_active: bool = Field(default=True, description="Whether flow is active")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SpecialFlow1CreateRequest(BaseModel):
    """Request to create a new Dialogue-Driven Flow"""
    name: str = Field(..., description="Flow name")
    description: Optional[str] = None
    config: SpecialFlow1Config = Field(..., description="Flow configuration")
    is_active: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SpecialFlow1UpdateRequest(BaseModel):
    """Request to update an existing Dialogue-Driven Flow"""
    name: str = Field(..., description="Flow name")
    description: Optional[str] = None
    config: SpecialFlow1Config = Field(..., description="Flow configuration")
    is_active: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SpecialFlow1ExecuteRequest(BaseModel):
    """Request to execute a Dialogue-Driven Flow"""
    initial_input: Optional[str] = Field(None, description="Optional initial input (for DB tool SQL or request params)")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    resume_from_phase: Optional[str] = Field(None, description="Resume from specific phase: 'dialogue_phase1', 'dialogue_phase2'. If provided, dialogue_phase1_result must also be provided.")
    dialogue_phase1_result: Optional[Dict[str, Any]] = Field(None, description="Dialogue phase 1 result to resume from (required if resume_from_phase is set)")
    dialogue_phase2_result: Optional[Dict[str, Any]] = Field(None, description="Dialogue phase 2 result to resume from (required if resume_from_phase is 'dialogue_phase2')")
    initial_data: Optional[Dict[str, Any]] = Field(None, description="Initial data from previous execution (required if resuming)")


class SpecialFlow1ExecuteResponse(BaseModel):
    """Response from executing a Dialogue-Driven Flow"""
    flow_id: str = Field(..., description="Flow ID")
    flow_name: str = Field(..., description="Flow name")
    success: bool = Field(..., description="Whether flow completed successfully")
    phase: str = Field(..., description="Current phase: 'initial_data', 'dialogue_phase1', 'data_fetch', 'dialogue_phase2', 'final_processing', 'api_call', 'complete'")
    initial_data: Optional[Dict[str, Any]] = Field(None, description="Initial data fetched")
    dialogue_phase1: Optional[Dict[str, Any]] = Field(None, description="Dialogue phase 1 result")
    fetched_data: Optional[Dict[str, Any]] = Field(None, description="Mid-dialogue fetched data")
    dialogue_phase2: Optional[Dict[str, Any]] = Field(None, description="Dialogue phase 2 result")
    final_outcome: Optional[str] = Field(None, description="Final processing outcome")
    api_call_result: Optional[Dict[str, Any]] = Field(None, description="Final API call result")
    total_execution_time: float = Field(..., description="Total execution time in seconds")
    error: Optional[str] = Field(None, description="Error message if flow failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# Conversation Models (Multi-AI Conversation Module)
class ModelConversationConfig(BaseModel):
    """Configuration for a single AI model in the conversation"""
    provider: LLMProviderType = Field(..., description="LLM provider for this model")
    model_name: str = Field(..., description="Model name for this model")
    system_prompt: str = Field(..., description="System prompt for this model")
    rag_collection: Optional[str] = Field(None, description="Optional RAG collection to use for this model")


class ConversationConfig(BaseModel):
    """Configuration for a conversation between two AI models"""
    model1_config: ModelConversationConfig = Field(..., description="Configuration for first AI model")
    model2_config: ModelConversationConfig = Field(..., description="Configuration for second AI model")
    max_turns: int = Field(default=10, ge=5, le=100, description="Maximum number of turns (must be between 5 and 100)")


class ConversationCreateRequest(BaseModel):
    """Request to create a conversation configuration"""
    name: str = Field(..., description="Name for this conversation configuration")
    description: Optional[str] = Field(None, description="Description of the conversation")
    config: ConversationConfig = Field(..., description="Conversation configuration")


class ConversationStartRequest(BaseModel):
    """Request to start a new conversation session"""
    config_id: str = Field(..., description="Conversation configuration ID")
    topic: str = Field(..., description="Initial topic or prompt to start the conversation")


class ConversationMessage(BaseModel):
    """A single message in the conversation"""
    role: str = Field(..., description="Role: 'user', 'model1', or 'model2'")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="ISO timestamp of the message")
    turn_number: int = Field(..., description="Turn number in the conversation")


class ConversationTurnRequest(BaseModel):
    """Request to continue a conversation turn"""
    session_id: str = Field(..., description="Session ID from conversation start")
    user_message: Optional[str] = Field(None, description="Optional user message to inject into conversation")


class ConversationResponse(BaseModel):
    """Response from a conversation turn"""
    session_id: str = Field(..., description="Unique session ID")
    turn_number: int = Field(..., description="Current turn number")
    max_turns: int = Field(..., description="Maximum turns allowed")
    is_complete: bool = Field(..., description="Whether conversation has reached max turns")
    messages: List[ConversationMessage] = Field(..., description="Messages from this turn")
    conversation_history: List[ConversationMessage] = Field(..., description="Full conversation history")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")


class ConversationHistoryResponse(BaseModel):
    """Response for retrieving conversation history"""
    session_id: str = Field(..., description="Session ID")
    config_id: str = Field(..., description="Configuration ID used")
    config_name: str = Field(..., description="Configuration name")
    started_at: str = Field(..., description="ISO timestamp when conversation started")
    ended_at: Optional[str] = Field(None, description="ISO timestamp when conversation ended")
    total_turns: int = Field(..., description="Total number of turns")
    conversation_history: List[ConversationMessage] = Field(..., description="Full conversation history")
    saved_file_path: Optional[str] = Field(None, description="Path to saved conversation file")
