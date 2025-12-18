import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
import json
import logging
from typing import List, Dict, Any, Optional
from .config import settings
from .models import RAGDataInput, RAGDataValidation, DataFormat
import re


class RAGSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False
            )
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        self.collections = {}

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