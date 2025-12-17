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
