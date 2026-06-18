from langchain_ollama import OllamaLLM
from src.config import settings
from src.vector_store import get_local_retriever
import time
import json
from datetime import datetime
import os
from src.hybrid_retriever import hybrid_retrieve

def verify_response_safety(context: str, answer: str) -> bool:
    """
    Placeholder guardrail function.
    Future implementation:
    - Hallucination detection
    - PII checks
    - Source-grounding validation
    """
    return True


def execute_private_query(
    user_question: str,
    chat_history=None
):
    start_time = time.time()

    print(f"\nReceived Question: {user_question}")

    llm = OllamaLLM(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.LOCAL_LLM_MODEL
    )

    retriever = get_local_retriever()

    # -------------------------
    # Retrieval
    # -------------------------
    retrieval_start = time.time()

    retrieved_docs = hybrid_retrieve(user_question)

    retrieval_time = round(
        time.time() - retrieval_start,
        2
    )

    print(
        f"\nRetrieval Time: {retrieval_time}s"
    )

    print("\n===== INITIAL RETRIEVAL =====")

    for idx, doc in enumerate(retrieved_docs):
        print(f"\nDocument {idx + 1}")
        print(doc.page_content[:300])

    # -------------------------
    # Simple reranking
    # -------------------------
    from src.reranker import (
    rerank_documents
    )

    reranked_docs = rerank_documents(
        query=user_question,
        retrieved_docs=retrieved_docs,
        top_k=2
    )

    print("\n===== RETRIEVAL STATS =====")
    print("Retrieved Docs:", len(retrieved_docs))
    print("Reranked Docs:", len(reranked_docs))

    print("\n===== RERANKED DOCUMENTS =====")

    for idx, doc in enumerate(reranked_docs):
        print(f"\nTop Document {idx + 1}")
        print(doc.page_content[:300])

    # -------------------------
    # Build Context
    # -------------------------
    context = "\n\n".join(
        [
            doc.page_content[:400]
            for doc in reranked_docs
        ]
    )

    # -------------------------
    # Build Conversation Memory
    # -------------------------
    conversation_context = ""

    if chat_history:

        recent_history = chat_history[-5:]

        for turn in recent_history:

            conversation_context += (
                f"User: {turn['question']}\n"
                f"Assistant: {turn['answer']}\n\n"
            )

    # -------------------------
    # Prompt
    # -------------------------
    prompt = f"""
You are a secure corporate assistant.

Use the conversation history and retrieved documents.

Answer ONLY using the provided context.

If the answer is not available in the context say:

"I do not know based on the provided documents."

Conversation History:
{conversation_context}

Retrieved Context:
{context}

Current Question:
{user_question}

Answer:
"""

    llm_start = time.time()

    response = llm.invoke(prompt)

    llm_time = round(
        time.time() - llm_start,
        2
    )

    print(
        f"\nLLM Time: {llm_time}s"
    )

    print("\n===== FINAL ANSWER =====")
    print(response)

    # -------------------------
    # Sources
    # -------------------------
    sources = []

    for doc in reranked_docs:

        sources.append(
            {
                "document": os.path.basename(
                    doc.metadata.get("source", "")
                ),
                "page": doc.metadata.get(
                    "page",
                    "Unknown"
                ),
                "snippet": doc.page_content[:300]
            }
        )

    elapsed = round(
        time.time() - start_time,
        2
    )

    print(
        f"\nQuery completed in {elapsed} seconds"
    )

    os.makedirs("logs", exist_ok=True)

    with open(
        "logs/query_logs.jsonl",
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                {
                    "timestamp": str(datetime.now()),
                    "question": user_question,
                    "answer_length": len(response),
                    "retrieved_docs": len(retrieved_docs),
                    "reranked_docs": len(reranked_docs),
                    "retrieval_time": retrieval_time,
                    "llm_time": llm_time,
                    "total_time": elapsed
                }
            )
            + "\n"
        )

    return {
        "answer": response,
        "sources": sources,
        "metrics": {
            "latency_seconds": elapsed,
            "retrieval_seconds": retrieval_time,
            "llm_seconds": llm_time,
            "retrieved_docs": len(retrieved_docs),
            "reranked_docs": len(reranked_docs)
        }
    }