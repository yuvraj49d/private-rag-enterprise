from langchain_community.documents_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings
import os

def load_and_chunk_documents():
    """Load PDFs from local and split them securely"""
    if not os.path.exists(settings.DATA_DIR):
        os.makedirs(settings.DATA_DIR)
        print(f"Created {settings.DATA_DIR}. Drop the PDFs here")
        return []
    
    # Strict local parsing
    loader = DirectoryLoader(
        settings.DATA_DIR,
        glob="**/*.pdf",
        loader_cls=UnstructuredPDFLoader
    )

    documents = loader.load()

    #Splitting text but preserving sentence context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)
    return chunks