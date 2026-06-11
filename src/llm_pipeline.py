from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.config import settings
from src.vector_store import get_local_retriever

def verify_response_safety(context: str, answer: str) -> bool:
    """A guardrail function that acts as a local evaluator to stop hallucinations."""
    # You use a highly specific prompt to validate the answer against the source text
    # If the Local evaluator detects data that wasn't in the context, it flags it.
    pass

def execute_private_query(user_question: str) -> str:
    """Asks the local offline LLM a question using retrieved private context."""
    
    # Connects to Locally hosted Ollama server instance (No data sent to OpenAI/Anthropic)
    llm = Ollama (base_url=settings.OLLAMA_BASE_URL, model=settings.LOCAL_LLM_MODEL)
    retriever = get_local_retriever()

    # System prompt enforcing zero information leaks
    system_prompt = (
        "You are a secure corporate assistant. Use only the following pieces of retrieve context "
        "to answer the question. If you do not know the answer, say you do not know. "
        "Do not invent facts.\n\n"
        "Context: \n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Creating the secure pipeline execution chain

    question_answer_chain = create_stuff_documents_chain (1lm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": user_question})
    return response ["answer"]
    