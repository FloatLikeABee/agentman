import uvicorn
import asyncio
import logging
from src.api import app
from src.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Ground Control with MCP support...")
    
    # Run the FastAPI application
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug_mode,
        log_level="info"
    )



# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from services.document_service import DocumentService
# from services.retrieval_service import RetrievalService
# from services.llm_service import LLMService
# from models import DocumentInput, QueryRequest, QueryResponse
# import time

# app = FastAPI(title="RAG API", description="A production-ready RAG system with categorization and API control")

# # Initialize services
# document_service = DocumentService()
# retrieval_service = RetrievalService()
# llm_service = LLMService()

# # Load documents and initialize retriever
# @app.on_event("startup")
# async def startup_event():
#     print("Loading documents...")
#     docs = document_service.load_and_categorize_documents()
#     print(f"Loaded {len(docs)} documents.")
#     retrieval_service.initialize_retriever(k=3)

# @app.get("/")
# async def root():
#     return {"message": "Welcome to the RAG API"}

# @app.post("/query", response_model=QueryResponse)
# async def query_rag(request: QueryRequest):
#     try:
#         start_time = time.time()
#         retrieved_docs = retrieval_service.retrieve(request.query, k=request.k)
#         context = "\n".join([doc.page_content for doc in retrieved_docs])
#         answer = llm_service.generate_answer(context, request.query)
#         end_time = time.time()

#         sources = [doc.metadata["source"] for doc in retrieved_docs]
#         return QueryResponse(
#             answer=answer,
#             sources=sources,
#             time_taken=end_time - start_time
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/add_document")
# async def add_document(document: DocumentInput):
#     try:
#         doc = Document( # type: ignore
#             page_content=document.content,
#             metadata={
#                 "source": "manual_upload",
#                 "category": document.category.value,
#                 "title": document.metadata.get("title", "Untitled")
#             }
#         )
#         document_service.data_dir.mkdir(exist_ok=True)
#         with open(document_service.data_dir / f"{document.category.value}_{time.time()}.txt", "w") as f:
#             f.write(document.content)
#         return {"status": "Document added", "category": document.category.value}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

