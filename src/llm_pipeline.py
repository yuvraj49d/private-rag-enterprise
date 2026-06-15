from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.vector_store import get_local_retriever


def verify_response_safety(context: str, answer: str) -> bool:
    """
    Placeholder guardrail function.
    Future implementation:
    - Hallucination detection
    - PII checks
    - Source-grounding validation
    """
    return True


def execute_private_query(user_question: str) -> str:

    print(f"\nReceived Question: {user_question}")

    llm = Ollama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.LOCAL_LLM_MODEL,
    )

    retriever = get_local_retriever()

    # Stage 1 Retrieval
    retrieved_docs = retriever.invoke(user_question)

    print("\n===== INITIAL RETRIEVAL =====")

    for idx, doc in enumerate(retrieved_docs):
        print(f"\nDocument {idx + 1}")
        print(doc.page_content[:300])

    # Stage 2 Reranking
    from src.reranker import rerank_documents

    reranked_docs = rerank_documents(
        user_question,
        retrieved_docs,
        top_k=2
    )

    print("\n===== RERANKED DOCUMENTS =====")

    for idx, doc in enumerate(reranked_docs):
        print(f"\nTop Document {idx + 1}")
        print(doc.page_content[:300])

    context = "\n\n".join(
        [doc.page_content for doc in reranked_docs]
    )

    prompt = f"""
You are a secure corporate assistant.

Use ONLY the provided context.

If the answer is not in the context,
say:

"I do not know based on the provided documents."

Context:
{context}

Question:
{user_question}
"""

    response = llm.invoke(prompt)

    print("\n===== FINAL ANSWER =====")
    print(response)

    return response