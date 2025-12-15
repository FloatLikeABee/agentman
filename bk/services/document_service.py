from typing import List
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from utils.categorizer import categorize_document
from pathlib import Path

class DocumentService:
    def __init__(self):
        self.data_dir = Path("data/documents")
        self.data_dir.mkdir(exist_ok=True)

    def load_and_categorize_documents(self) -> List[Document]:
        loader = DirectoryLoader(str(self.data_dir), glob="**/*.txt")
        documents = loader.load()

        categorized_docs = []
        for doc in documents:
            category = categorize_document(doc.page_content)
            metadata = {
                "source": doc.metadata.get("source", "unknown"),
                "category": category.value,
                "title": doc.metadata.get("title", "Untitled")
            }
            doc.metadata.update(metadata)
            categorized_docs.append(doc)

        return categorized_docs