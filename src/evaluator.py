from ragas.metrics import Faithfulness, AnswerRelevance, ContextPrecision
from ragas import evaluate
from datasets import Dataset
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import settings
from src.llm_pipeline import execute_private_query
from src.vector_store import get_local_retriever
from src.logger import logger

def run_offline_ragas_evaluation():
    """Runs an automated data quality evaluation over a golden test dataset."""
    logger.info("Initializing offline Ragas evaluation suite...")

    # Using the local infrastructure as the evaluator judge
    evaluator_llm = Ollama (base_url=settings.OLLAMA_BASE_URL, model=settings.LOCAL_LLM_MODEL)
    evaluator_embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

    # Define a representative 'Golden Test Set' to stress-test your RAG code structure
    test_queries = [
        "What is the company compliance policy regarding data leaks?",
        "What are the parameters for remote network server security?"
    ]

    # Labeled ground truths (what a perfect human answer should Look Like)
    ground_truths = [
        "Company policy strictly mandates all sensitive data remain within air-gapped on-premise servers.",
        "Remote network access requires multi-factor authentication and hardware key configurations."
    ]

    generated_answers = []
    retrieved_contexts = []

    # Collect outputs from your active pipelines
    retriever = get_local_retriever()

    for query in test_queries:
        logger.info(f"Evaluating pipeline response for query: '{query}'")
        
        # Pull raw answers
        ans = execute_private_query(query)
        generated_answers.append(ans)
        
        # Pull what chunks your system grabbed
        docs = retriever.invoke(query)
        contexts = [doc.page_content for doc in docs]
        retrieved_contexts.append(contexts)

    # Format data structure cleanly into standard Hugging Face dataset format
    data_dict = {
        "user_input": test_queries,
        "response": generated_answers,
        "retrieved_contexts": retrieved_contexts,
        "reference": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)

    # Define the exact industry evaluation metrics to score
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevance(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm, embeddings=evaluator_embeddings)
    ]

    logger.info("Executing LLM-as-a-judge statistical processing loop...")
    results = evaluate(dataset=dataset, metrics=metrics)

    logger.info(f"Evaluation metrics calculated successfully: {results}")
    return results
    