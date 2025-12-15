from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import settings
from typing import List

class VectorStoreRepository:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store = Chroma(
            persist_directory=settings.persist_directory,
            embedding_function=self.embeddings
        )

    def add_documents(self, documents: List[Document]):
        self.vector_store.add_documents(documents)

    def get_retriever(self, k: int = 3):
        return self.vector_store.as_retriever(search_kwargs={"k": k})