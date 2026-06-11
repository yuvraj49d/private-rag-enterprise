from sentence_transformers import CrossEncoder
import numpy as np

def rerank_documents (query: str, retrieved_docs: list) -> list:
    """Uses a local Cross-Encoder to re-rank documents for higher accuracy."""
    # Using an extremely accurate, open-source local re-ranker
    model = CrossEncoder("BAAI/bge-reranker-base")
    
    # Pair the query with each document text
    pairs = [[query, doc.page_content] for doc in retrieved_docs]
    scores = model.predict(pairs)

    # Sort documents by theur nrlew relevance scores
    ranked_indices = np.argsort(scores)[::-1]
    reranked_docs = [retrieved_docs[i] for i in ranked_indices]

    # Return only the top 2 highest scoring contexts
    return reranked_docs[:2]