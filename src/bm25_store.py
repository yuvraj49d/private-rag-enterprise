from rank_bm25 import BM25Okapi
from src.ingest import load_and_chunk_documents

bm25 = None
all_docs = None


def initialize_bm25():

    global bm25
    global all_docs

    if bm25 is not None:
        return bm25, all_docs

    print("\nInitializing BM25 Index...")

    all_docs = load_and_chunk_documents()

    corpus = [
        doc.page_content
        for doc in all_docs
    ]

    tokenized = [
        doc.split()
        for doc in corpus
    ]

    bm25 = BM25Okapi(tokenized)

    print(
        f"BM25 Ready | Documents: {len(all_docs)}"
    )

    return bm25, all_docs