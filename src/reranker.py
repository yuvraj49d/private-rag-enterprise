from sentence_transformers import CrossEncoder
import numpy as np

# Load once globally
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_documents(
    query: str,
    retrieved_docs: list,
    top_k: int = 3
):

    if not retrieved_docs:
        return []

    pairs = [
        [query, doc.page_content]
        for doc in retrieved_docs
    ]

    scores = reranker_model.predict(
        pairs
    )

    ranked_indices = np.argsort(
        scores
    )[::-1]

    reranked_docs = [
        retrieved_docs[i]
        for i in ranked_indices
    ]

    return reranked_docs[:top_k]