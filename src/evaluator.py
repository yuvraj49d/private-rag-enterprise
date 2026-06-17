from datasets import Dataset
from ragas import evaluate
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)

from langchain_ollama import ChatOllama

from src.llm_pipeline import execute_private_query
from src.vector_store import get_local_retriever


def run_offline_ragas_evaluation():

    # RAGAS Judge Model
    judge_llm = ChatOllama(
        model="qwen2.5:1.5b",
        base_url="http://localhost:11434"
    )

    local_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
    )

    test_queries = [
        "What is Artificial Intelligence?",
        "What are the benefits of it?",
    ]

    ground_truths = [
        "AI is the ability of computers to exhibit intelligent behaviour.",
        "AI improves efficiency, personalization, forecasting and customer service."
    ]

    generated_answers = []
    retrieved_contexts = []

    retriever = get_local_retriever()

    for query in test_queries:

        result = execute_private_query(query)

        generated_answers.append(
            result["answer"]
        )

        docs = retriever.invoke(query)

        retrieved_contexts.append(
            [
                doc.page_content
                for doc in docs
            ]
        )

    dataset = Dataset.from_dict(
        {
            "question": test_queries,
            "answer": generated_answers,
            "contexts": retrieved_contexts,
            "ground_truth": ground_truths
        }
    )

    print("\n===== STARTING RAGAS EVALUATION =====")

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision
        ],
        llm=judge_llm,
        embeddings=local_embeddings
    )

    print("\n===== RAGAS RESULTS =====")
    print(results)

    return results