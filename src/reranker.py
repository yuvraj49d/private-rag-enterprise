from sentence_transformers import CrossEncoder
import numpy as np

# Load once globally
reranker_model = CrossEncoder("BAAI/bge-reranker-base")


def rerank_documents(query: str, retrieved_docs: list, top_k: int = 2):
    """
    Re-rank retrieved documents using a CrossEncoder.
    """

    if not retrieved_docs:
        return []

    pairs = [
        [query, doc.page_content]
        for doc in retrieved_docs
    ]

    scores = reranker_model.predict(pairs)

    ranked_indices = np.argsort(scores)[::-1]

    reranked_docs = [
        retrieved_docs[i]
        for i in ranked_indices
    ]

    return reranked_docs[:top_k]