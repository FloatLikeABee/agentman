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
from langchain_community.embeddings import HuggingFaceEmbeddings
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
