from typing import Optional
from langchain_core.retrievers import BaseRetriever
from config import settings
from repositories.vector_store import VectorStoreRepository

class RetrievalService:
    def __init__(self):
        self.vector_store = VectorStoreRepository()
        self.retriever: Optional[BaseRetriever] = None

    def initialize_retriever(self, k: int = 3):
        self.retriever = self.vector_store.get_retriever(k)

    def retrieve(self, query: str, k: int = 3) -> list:
        if not self.retriever:
            raise ValueError("Retriever not initialized. Call initialize_retriever first.")
        return self.retriever.invoke(query)