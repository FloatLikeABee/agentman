import os
import logging
import ssl
import time
from functools import wraps
from typing import Optional

# Disable ChromaDB telemetry BEFORE importing chromadb
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "True"

# Configure HuggingFace Hub SSL, proxy, mirror and retry settings
# Note: Settings will be loaded after imports, so we check both env vars and will update after settings load
def configure_hf_ssl(ssl_verify: bool = True, timeout: int = 300, proxy: Optional[str] = None, mirror: Optional[str] = None):
    """Configure HuggingFace Hub SSL, proxy, mirror and timeout settings"""
    if not ssl_verify:
        os.environ["CURL_CA_BUNDLE"] = ""
        os.environ["REQUESTS_CA_BUNDLE"] = ""
        # Create unverified SSL context for HuggingFace Hub
        ssl._create_default_https_context = ssl._create_unverified_context
    
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout)
    os.environ["HF_HUB_CACHE"] = os.getenv("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface"))
    
    # Configure proxy if provided
    if proxy:
        # HuggingFace Hub uses standard HTTP_PROXY and HTTPS_PROXY environment variables
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        # Also set for requests library
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy
        logging.info(f"HuggingFace proxy configured: {proxy}")
    
    # Configure mirror if provided (useful for China users)
    if mirror:
        # HF_ENDPOINT is used by huggingface_hub to set a custom endpoint
        os.environ["HF_ENDPOINT"] = mirror
        logging.info(f"HuggingFace mirror configured: {mirror}")

# Initial configuration (will be updated after settings load)
configure_hf_ssl()

# Suppress ChromaDB telemetry errors before import
chromadb_telemetry_logger = logging.getLogger("chromadb.telemetry")
chromadb_telemetry_logger.setLevel(logging.CRITICAL)
chromadb_telemetry_logger.disabled = True

# Suppress PostHog telemetry errors
posthog_logger = logging.getLogger("chromadb.telemetry.product.posthog")
posthog_logger.setLevel(logging.CRITICAL)
posthog_logger.disabled = True

# Suppress HuggingFace Hub SSL warnings if SSL verification is disabled (will be updated after settings load)
def suppress_hf_warnings():
    """Suppress HuggingFace Hub SSL warnings if SSL verification is disabled"""
    if os.getenv("HF_SSL_VERIFY", "true").lower() == "false":
        huggingface_logger = logging.getLogger("huggingface_hub.utils._http")
        huggingface_logger.setLevel(logging.ERROR)

suppress_hf_warnings()

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # deprecated fallback
import pandas as pd
import json
from typing import List, Dict, Any, Optional
from .config import settings
from .models import RAGDataInput, RAGDataValidation, DataFormat
import re


def retry_with_backoff(max_retries=3, initial_delay=2, backoff_factor=2):
    """Decorator to retry function calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (backoff_factor ** attempt)
                        logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
            raise last_exception
        return wrapper
    return decorator


class RAGSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configure HuggingFace settings from config
        ssl_verify = getattr(settings, 'hf_ssl_verify', True)
        timeout = getattr(settings, 'hf_download_timeout', 300)
        proxy = getattr(settings, 'hf_proxy', None)
        mirror = getattr(settings, 'hf_mirror', None)
        configure_hf_ssl(ssl_verify=ssl_verify, timeout=timeout, proxy=proxy, mirror=mirror)
        
        if proxy:
            self.logger.info(f"Using HuggingFace proxy: {proxy}")
        if mirror:
            self.logger.info(f"Using HuggingFace mirror: {mirror}")
        
        # Suppress HuggingFace Hub SSL warnings if SSL verification is disabled
        if not ssl_verify:
            suppress_hf_warnings()
        
        # Initialize embeddings with retry logic
        self.embeddings = self._initialize_embeddings()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        self.collections = {}
    
    @retry_with_backoff(max_retries=5, initial_delay=3, backoff_factor=2)
    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings with retry logic"""
        try:
            self.logger.info(f"Initializing embeddings model: {settings.embedding_model}")
            if not getattr(settings, 'hf_ssl_verify', True):
                self.logger.warning("SSL verification is disabled for HuggingFace downloads (development mode)")
            
            embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={'device': 'cpu'},
                # Add cache configuration
                cache_folder=os.getenv("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface"))
            )
            self.logger.info("Embeddings model initialized successfully")
            return embeddings
        except Exception as e:
            self.logger.error(f"Failed to initialize embeddings: {e}")
            # Provide helpful suggestions based on error type
            error_str = str(e).lower()
            if "timeout" in error_str or "connection" in error_str or "maxretry" in error_str:
                self.logger.warning(
                    "Connection timeout/error detected. To fix this, you can:\n"
                    "1. Use a HuggingFace mirror (recommended for China users):\n"
                    "   - Add to .env: HF_MIRROR=https://hf-mirror.com\n"
                    "   - Or set environment variable: HF_MIRROR=https://hf-mirror.com\n"
                    "2. Use a proxy server:\n"
                    "   - Add to .env: HF_PROXY=http://your-proxy:port\n"
                    "   - Or set environment variable: HF_PROXY=http://your-proxy:port\n"
                    "3. Increase timeout:\n"
                    "   - Add to .env: HF_DOWNLOAD_TIMEOUT=600\n"
                    "4. Check your network/firewall settings\n"
                    "5. Try downloading the model manually and placing it in the cache folder"
                )
            elif "SSL" in str(e) or "SSL" in str(type(e)) or "SSLError" in str(type(e)):
                self.logger.warning(
                    "SSL error detected. To fix this, you can:\n"
                    "1. Set environment variable: HF_SSL_VERIFY=false (development only)\n"
                    "2. Or add to .env file: HF_SSL_VERIFY=false\n"
                    "3. Check your network/firewall settings\n"
                    "4. Update SSL certificates on your system"
                )
            raise

    def validate_data(self, data_input: RAGDataInput) -> RAGDataValidation:
        """Validate RAG data input"""
        errors = []
        warnings = []
        
        try:
            if data_input.format == DataFormat.JSON:
                # Validate JSON
                try:
                    json_data = json.loads(data_input.content)
                    if isinstance(json_data, list):
                        record_count = len(json_data)
                    elif isinstance(json_data, dict):
                        record_count = 1
                    else:
                        errors.append("JSON must be an object or array")
                        record_count = None
                except json.JSONDecodeError as e:
                    errors.append(f"Invalid JSON: {str(e)}")
                    record_count = None
                    
            elif data_input.format == DataFormat.CSV:
                # Validate CSV
                try:
                    df = pd.read_csv(pd.StringIO(data_input.content))
                    record_count = len(df)
                    if record_count == 0:
                        warnings.append("CSV file is empty")
                except Exception as e:
                    errors.append(f"Invalid CSV: {str(e)}")
                    record_count = None
                    
            elif data_input.format == DataFormat.TXT:
                # Validate text
                if not data_input.content.strip():
                    errors.append("Text content is empty")
                    record_count = None
                else:
                    record_count = 1
                    
            else:
                errors.append(f"Unsupported format: {data_input.format}")
                record_count = None
                
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            record_count = None
            
        is_valid = len(errors) == 0
        
        return RAGDataValidation(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            record_count=record_count
        )

    def process_data(self, data_input: RAGDataInput) -> List[str]:
        """Process and split data into chunks"""
        if data_input.format == DataFormat.JSON:
            # Parse JSON and convert to text
            json_data = json.loads(data_input.content)
            if isinstance(json_data, list):
                texts = []
                for item in json_data:
                    texts.append(json.dumps(item, indent=2))
                return texts
            else:
                return [json.dumps(json_data, indent=2)]
                
        elif data_input.format == DataFormat.CSV:
            # Convert CSV to text
            df = pd.read_csv(pd.StringIO(data_input.content))
            texts = []
            for _, row in df.iterrows():
                texts.append(row.to_string())
            return texts
            
        elif data_input.format == DataFormat.TXT:
            # Split text into chunks
            return self.text_splitter.split_text(data_input.content)
            
        else:
            raise ValueError(f"Unsupported format: {data_input.format}")

    @staticmethod
    def sanitize_collection_name(name: str) -> str:
        """Sanitize collection name for ChromaDB: 3-63 chars, alphanumeric/underscore/hyphen only, no spaces."""
        if not name or not isinstance(name, str):
            return ""
        s = name.strip()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-zA-Z0-9_-]", "_", s)
        s = re.sub(r"_+", "_", s)
        s = s.strip("_-")
        if not s or len(s) < 3:
            return ""
        if len(s) > 63:
            s = s[:63].rstrip("_-")
        return s

    def create_collection(self, name: str, description: str = "") -> str:
        """Create a new RAG collection"""
        try:
            collection = self.chroma_client.create_collection(
                name=name,
                metadata={"description": description}
            )
            self.collections[name] = collection
            self.logger.info(f"Created collection: {name}")
            return name
        except Exception as e:
            self.logger.error(f"Error creating collection {name}: {e}")
            raise

    def add_data_to_collection(self, collection_name: str, data_input: RAGDataInput) -> bool:
        """Add data to a RAG collection"""
        try:
            # Validate data first
            validation = self.validate_data(data_input)
            if not validation.is_valid:
                self.logger.error(f"Data validation failed: {validation.errors}")
                return False
            
            # Get or create collection
            if collection_name not in self.collections:
                try:
                    collection = self.chroma_client.get_collection(collection_name)
                    self.collections[collection_name] = collection
                except:
                    collection = self.create_collection(collection_name, data_input.description or "")
            
            # Process data into chunks
            texts = self.process_data(data_input)
            
            # Add to collection
            collection = self.collections[collection_name]
            collection.add(
                documents=texts,
                metadatas=[{
                    "source": data_input.name,
                    "format": data_input.format.value,
                    "tags": ",".join(data_input.tags),
                    **data_input.metadata
                }] * len(texts),
                ids=[f"{data_input.name}_{i}" for i in range(len(texts))]
            )
            
            self.logger.info(f"Added {len(texts)} documents to collection {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding data to collection {collection_name}: {e}")
            return False

    def query_collection(self, collection_name: str, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query a RAG collection"""
        try:
            if collection_name not in self.collections:
                collection = self.chroma_client.get_collection(collection_name)
                self.collections[collection_name] = collection
            
            collection = self.collections[collection_name]
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else None
                    })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error querying collection {collection_name}: {e}")
            return []

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all RAG collections"""
        try:
            collections = self.chroma_client.list_collections()
            return [
                {
                    'name': col.name,
                    'count': col.count(),
                    'metadata': col.metadata
                }
                for col in collections
            ]
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return []

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a RAG collection"""
        try:
            self.chroma_client.delete_collection(collection_name)
            if collection_name in self.collections:
                del self.collections[collection_name]
            self.logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting collection {collection_name}: {e}")
            return False

    async def smart_import(
        self,
        file_content: str,
        file_format: str,
        llm_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        processing_instructions: Optional[str] = None,
        auto_name: bool = True,
    ) -> Dict[str, Any]:
        """
        Smart Import: Import CSV/JSON file, process with AI, auto-name, and save to RAG.
        
        Steps:
        1. Parse the file (CSV or JSON)
        2. Use AI to clean/abstract/transform the data
        3. Generate a good collection name (if auto_name=True)
        4. Convert to RAG format
        5. Save to RAG system
        
        Returns:
            Dict with collection_name, description, processed_data, and metadata
        """
        from .config import settings
        from .llm_factory import LLMFactory, LLMProvider
        from .llm_langchain_wrapper import LangChainLLMWrapper
        from .models import LLMProviderType
        
        try:
            # Step 1: Parse the file
            self.logger.info(f"[Smart Import] Parsing {file_format} file...")
            if file_format.lower() == "csv":
                df = pd.read_csv(pd.StringIO(file_content))
                original_data = df.to_dict('records')
                original_count = len(original_data)
                # Convert to JSON string for AI processing
                raw_data_str = json.dumps(original_data, indent=2, default=str)
            elif file_format.lower() == "json":
                original_data = json.loads(file_content)
                if isinstance(original_data, list):
                    original_count = len(original_data)
                    raw_data_str = json.dumps(original_data, indent=2, default=str)
                elif isinstance(original_data, dict):
                    original_count = 1
                    raw_data_str = json.dumps(original_data, indent=2, default=str)
                else:
                    raise ValueError("JSON must be an object or array")
            else:
                raise ValueError(f"Unsupported file format: {file_format}. Supported: csv, json")
            
            # Step 2: Use AI to process the data
            self.logger.info(f"[Smart Import] Processing data with AI...")
            
            # Determine LLM provider
            if llm_provider:
                provider_type = LLMProviderType(llm_provider.lower())
            else:
                provider_type = LLMProviderType(settings.default_llm_provider.lower())
            
            if provider_type == LLMProviderType.GEMINI:
                api_key = settings.gemini_api_key
                model = model_name or settings.gemini_default_model
            elif provider_type == LLMProviderType.QWEN:
                api_key = settings.qwen_api_key
                model = model_name or settings.qwen_default_model
            elif provider_type == LLMProviderType.MISTRAL:
                api_key = getattr(settings, 'mistral_api_key', '')
                model = model_name or settings.mistral_default_model
            elif provider_type == LLMProviderType.GROQ:
                api_key = getattr(settings, 'groq_api_key', '')
                model = model_name or getattr(settings, 'groq_default_model', 'llama-3.3-70b-versatile')
            else:
                provider_type = LLMProviderType.GEMINI
                api_key = settings.gemini_api_key
                model = settings.gemini_default_model
            
            # Create LLM caller
            llm_caller = LLMFactory.create_caller(
                provider=LLMProvider(provider_type.value),
                api_key=api_key,
                model=model,
                temperature=0.3,  # Lower temperature for more consistent processing
                max_tokens=8192,
                timeout=settings.api_timeout,
            )
            llm = LangChainLLMWrapper(llm_caller=llm_caller)
            
            # Build processing prompt
            base_instructions = """You are a data processing assistant. Your task is to clean, abstract, and transform the provided data into a well-structured format suitable for RAG (Retrieval-Augmented Generation) systems.

Requirements:
1. Clean the data: Remove duplicates, fix inconsistencies, standardize formats
2. Abstract the data: Extract key information and create meaningful summaries
3. Structure for RAG: Format as JSON array where each item is a self-contained, searchable document
4. Preserve important information: Don't lose critical data points
5. Make it searchable: Ensure each document has clear, descriptive content

Output format: Return ONLY valid JSON array, no additional text or markdown."""
            
            processing_prompt = f"""{base_instructions}

{processing_instructions or ''}

Original Data:
{raw_data_str[:50000]}  # Limit to avoid token limits

Process and return the cleaned, abstracted data as a JSON array."""
            
            # Call AI for processing
            processed_data_str = await llm.ainvoke(processing_prompt)
            
            # Extract JSON from response (handle markdown code blocks)
            processed_data_str = processed_data_str.strip()
            if '```json' in processed_data_str:
                processed_data_str = processed_data_str.split('```json')[1].split('```')[0].strip()
            elif '```' in processed_data_str:
                processed_data_str = processed_data_str.split('```')[1].split('```')[0].strip()
            
            # Parse processed data
            try:
                processed_data = json.loads(processed_data_str)
                if not isinstance(processed_data, list):
                    processed_data = [processed_data]
                processed_count = len(processed_data)
            except json.JSONDecodeError as e:
                self.logger.error(f"[Smart Import] Failed to parse AI-processed data: {e}")
                # Fallback: use original data
                processed_data = original_data if isinstance(original_data, list) else [original_data]
                processed_count = len(processed_data)
            
            # Step 3: Generate collection name (if auto_name)
            collection_name = None
            collection_description = None
            
            if auto_name:
                self.logger.info(f"[Smart Import] Generating collection name with AI...")
                name_prompt = f"""Based on the following data, generate a short, descriptive name (2-4 words, lowercase with underscores) and a brief description (1-2 sentences) for a RAG knowledge base collection.

Data sample (first 3 items):
{json.dumps(processed_data[:3], indent=2, default=str)[:2000]}

Return JSON format:
{{
    "name": "descriptive_collection_name",
    "description": "Brief description of what this collection contains"
}}"""
                
                name_response = await llm.ainvoke(name_prompt)
                name_response = name_response.strip()
                if '```json' in name_response:
                    name_response = name_response.split('```json')[1].split('```')[0].strip()
                elif '```' in name_response:
                    name_response = name_response.split('```')[1].split('```')[0].strip()
                
                try:
                    name_data = json.loads(name_response)
                    collection_name = name_data.get("name", "imported_data")
                    collection_description = name_data.get("description", "Imported and processed data")
                except:
                    # Fallback name generation
                    collection_name = f"imported_{file_format}_{int(time.time())}"
                    collection_description = f"Imported {file_format} data processed with AI"
            else:
                collection_name = f"imported_{file_format}_{int(time.time())}"
                collection_description = f"Imported {file_format} data"
            
            # Ensure collection name is valid (lowercase, underscores, no spaces)
            collection_name = collection_name.lower().replace(" ", "_").replace("-", "_")
            collection_name = re.sub(r'[^a-z0-9_]', '', collection_name)
            if not collection_name:
                collection_name = f"imported_{int(time.time())}"
            
            # Step 4: Save to RAG system
            self.logger.info(f"[Smart Import] Saving to RAG collection: {collection_name}")
            
            rag_input = RAGDataInput(
                name=f"smart_import_{int(time.time())}",
                description=collection_description,
                format=DataFormat.JSON,
                content=json.dumps(processed_data, indent=2, default=str),
                tags=["smart_import", file_format, "ai_processed"],
                metadata={
                    "original_format": file_format,
                    "original_count": original_count,
                    "processed_count": processed_count,
                    "llm_provider": provider_type.value,
                    "model_used": model,
                    "imported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            
            success = self.add_data_to_collection(collection_name, rag_input)
            
            if success:
                return {
                    "success": True,
                    "collection_name": collection_name,
                    "collection_description": collection_description,
                    "processed_data": json.dumps(processed_data, indent=2, default=str),
                    "original_record_count": original_count,
                    "processed_record_count": processed_count,
                    "message": f"Successfully imported and processed {original_count} records into collection '{collection_name}'",
                    "metadata": {
                        "llm_provider": provider_type.value,
                        "model_used": model,
                        "processing_instructions": processing_instructions,
                    }
                }
            else:
                return {
                    "success": False,
                    "collection_name": collection_name,
                    "collection_description": collection_description,
                    "processed_data": None,
                    "original_record_count": original_count,
                    "processed_record_count": processed_count,
                    "message": "Failed to save processed data to RAG collection",
                    "metadata": {}
                }
                
        except Exception as e:
            self.logger.error(f"[Smart Import] Error: {e}", exc_info=True)
            return {
                "success": False,
                "collection_name": "",
                "collection_description": "",
                "processed_data": None,
                "original_record_count": None,
                "processed_record_count": None,
                "message": f"Error during smart import: {str(e)}",
                "metadata": {}
            }
