from rank_bm25 import BM25Okapi
from src.vector_store import get_local_retriever
from src.ingest import load_and_chunk_documents
from src.bm25_store import initialize_bm25

def hybrid_retrieve(
    query: str,
    vector_k: int = 4,
    bm25_k: int = 4
):
    """
    Hybrid Retrieval:
    1. Vector Search (Chroma)
    2. BM25 Keyword Search
    3. Merge Results
    """

    # --------------------------
    # Vector Search
    # --------------------------

    retriever = get_local_retriever()

    vector_docs = retriever.invoke(query)

    # --------------------------
    # BM25 Search
    # --------------------------

    bm25, all_docs = initialize_bm25()

    scores = bm25.get_scores(
        query.split()
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    bm25_docs = [
        all_docs[i]
        for i in ranked_indices[:bm25_k]
    ]

    # --------------------------
    # Merge + Deduplicate
    # --------------------------

    merged = vector_docs + bm25_docs

    unique_docs = []

    seen = set()

    for doc in merged:

        content = doc.page_content[:200]

        if content not in seen:

            seen.add(content)

            unique_docs.append(doc)

    return unique_docs