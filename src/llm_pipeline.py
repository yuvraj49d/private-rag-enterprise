from langchain_community.llms import Ollama
from src.config import settings
from src.vector_store import get_local_retriever
import time
import os

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

    llm = Ollama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.LOCAL_LLM_MODEL,
    )

    retriever = get_local_retriever()

    # -------------------------
    # Retrieval
    # -------------------------
    retrieved_docs = retriever.invoke(user_question)

    print("\n===== INITIAL RETRIEVAL =====")

    for idx, doc in enumerate(retrieved_docs):
        print(f"\nDocument {idx + 1}")
        print(doc.page_content[:300])

    # -------------------------
    # Simple reranking
    # -------------------------
    reranked_docs = retrieved_docs[:2]

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
        [doc.page_content for doc in reranked_docs]
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

    response = llm.invoke(prompt)

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

    return {
        "answer": response,
        "sources": sources,
        "metrics": {
            "latency_seconds": elapsed,
            "retrieved_docs": len(retrieved_docs),
            "reranked_docs": len(reranked_docs)
        }
    }