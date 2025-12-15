import os
import time
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


MODEL = "qwen3:4b"

# Load documents from a directory
def load_documents(directory):
    documents = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.endswith(".csv"):
            loader = CSVLoader(file_path)
        else:
            continue
        documents.extend(loader.load())
    return documents

# Split documents into chunks
documents = load_documents("./data/")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Adjust based on your needs
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Create vector store
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Load a local Ollama model (e.g., "llama2", "mistral")
llm = Ollama(
    model=MODEL,
    temperature=0.5,
    system="You are a helpful assistant plus a very high class problem solver, and you give high class digital supports. Answer questions using the provided context.",
)

# Initialize retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 chunks

# Build QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

def main():
    while True:
        print(f"You're chatting with: {MODEL} --")
        try:
            user_input = input("prompt: ").strip().lower()
            if user_input == "exit":
                print("Goodbye!")
                break
            else:
                start = time.ctime()
                start_count = time.time()
                print(start)
                result = qa_chain.invoke({"query": user_input})
                print(f"Answer: {result['result']}")
                end = time.ctime()
                end_count = time.time()
                print(end)
                print(f"time spent for response: {end_count - start_count}")
                print("\nSources:")
                for doc in result["source_documents"]:
                    print(doc.metadata["source"])  # Shows which file the chunk came from
        except ValueError:
            print("Error: Invalid number format.")
        except Exception as e:
            print(f"Error: {e}")

main()
