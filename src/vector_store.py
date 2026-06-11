from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import settings
from src.ingest import load_and_chunk_documents
from src.logger import logger

def build_local_vector_db():
    """Generates local embeddings and saves them to a secure disk database."""
    logger.info("Initializing local HuggingFace embeddings...")
    # Computes purely on Local CPU/GPU via PyTorch
    embeddings = HuggingFaceEmbeddings (model_name=settings.EMBEDDING_MODEL_NAME)

    chunks = load_and_chunk_documents()
    if not chunks:
        logger.error("No documents found to index.")
        return None

    logger.info(f"Indexing {len(chunks)} chunks into local ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.DB_DIR
    )

    logger.info(f"Database successfully saved locally at: {settings.DB_DIR}")
    return vector_db

def get_local_retriever():
    """Loads the pre-built local database for quick querying."""
    logger.info("Connecting to persistent local ChromaDB instance...")
    embeddings = HuggingFaceEmbeddings (model_name=settings.EMBEDDING_MODEL_NAME)
    vector_db = Chroma (
        persist_directory=settings.DB_DIR,
        embedding_function=embeddings

    )
    return vector_db.as_retriever(search_kwargs={"k": 3})