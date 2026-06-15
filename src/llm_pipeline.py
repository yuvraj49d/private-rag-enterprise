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
    """
    Executes a secure RAG query against locally stored enterprise documents.
    """

    print(f"\nReceived Question: {user_question}")

    # Connect to local Ollama instance
    llm = Ollama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.LOCAL_LLM_MODEL,
    )

    retriever = get_local_retriever()

    # Debug retrieval output
    try:
        docs = retriever.invoke(user_question)

        print("\n===== RETRIEVED DOCUMENTS =====")

        if not docs:
            print("No documents retrieved!")

        for idx, doc in enumerate(docs):
            print(f"\n--- Document {idx + 1} ---")
            print(doc.page_content[:1000])

    except Exception as e:
        print(f"Retriever Error: {e}")
        raise

    # System prompt
    system_prompt = """
    You are a secure corporate assistant.

    Use ONLY the provided context to answer the user's question.

    Rules:
    - Do not invent facts.
    - Do not use outside knowledge.
    - If the answer is not present in the context, say:
      "I do not know based on the provided documents."

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create document QA chain
    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    # Create RAG chain
    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    try:
        response = rag_chain.invoke(
            {
                "input": user_question
            }
        )

        print("\n===== MODEL RESPONSE =====")
        print(response)

        answer = response.get(
            "answer",
            "No answer generated."
        )

        return answer

    except Exception as e:
        print(f"RAG Chain Error: {e}")
        raise