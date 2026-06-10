from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import settings
from src.ingest import load_and_chunk_documents

def build_local_vector_db():
    """Generates local embeddings and saves them to a secure disk database."""
    print("Initializing local HuggingFace embeddings...")
    # Computes purely on Local CPU/GPU via PyTorch
    embeddings = HuggingFaceEmbeddings (model_name=settings.EMBEDDING_MODEL_NAME)

    chunks = load_and_chunk_documents()
    if not chunks:
        print("No documents found to index.")
        return None

    print(f"Indexing {len(chunks)} chunks into local ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.DB_DIR
    )

    print(f"Database successfully saved locally at: {settings.DB_DIR}")
    return vector_db

def get_local_retriever():
    """Loads the pre-built local database for quick querying."""
    embeddings = HuggingFaceEmbeddings (model_name=settings.EMBEDDING_MODEL_NAME)
    vector_db = Chroma (
        persist_directory=settings.DB_DIR,
        embedding_function=embeddings

    )
    return vector_db.as_retriever(search_kwargs={"k": 3})