import sys
import os
import json

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import streamlit as st

from src.llm_pipeline import execute_private_query
from src.vector_store import build_local_vector_db

os.makedirs("data", exist_ok=True)

st.set_page_config(page_title="Enterprise Private RAG", layout="wide")

with st.sidebar:
    st.header("System Status")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.info("Model: qwen2.5:1.5b")
    st.info("Embedding: BAAI/bge-small-en-v1.5")

    st.markdown("---")
    st.subheader("Analytics")

    log_count = 0

    if os.path.exists("logs/query_logs.jsonl"):

        with open(
            "logs/query_logs.jsonl",
            "r",
            encoding="utf-8"
        ) as f:

            log_count = len(f.readlines())

    st.metric(
        "Total Queries",
        log_count
    )

    st.success("Vector Database: Connected")

st.title("Secure Enterprise Offline RAG Control Panel")
st.subheader("Air-gapped, zero data-leak document intelligence")

# Create functional columns Layout

col1, col2 = st.columns([2, 1])

with col1:

    st.markdown("### Chat with Secure Documents")

    # Initialize basic chat session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): 
            st.markdown(msg["content"])

    # Handle user text input
    if user_query := st.chat_input("Ask a question about internal corporate PDFs..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Process via our backend pipeline Layout
        with st.spinner("Retrieving local context and generating response..."):
            try:
                result = execute_private_query(
                    user_query,
                    st.session_state.get(
                        "chat_history",
                        []
                    )
                )

                answer = result["answer"]
                sources = result["sources"]
                metrics = result.get("metrics", {})

                with st.chat_message("assistant"):

                    st.markdown(answer)

                    metrics = result.get("metrics", {})

                    if metrics:
                        st.caption(
                            f"Total: {metrics['latency_seconds']}s | "
                            f"Retrieval: {metrics['retrieval_seconds']}s | "
                            f"LLM: {metrics['llm_seconds']}s | "
                            f"Retrieved: {metrics['retrieved_docs']} | "
                            f"Used: {metrics['reranked_docs']}"
                        )

                    with st.expander("View Sources"):

                        for source in sources:

                            st.markdown(
                                f"### {source['document']}"
                            )

                            st.markdown(
                                f"**Page:** {source['page']}"
                            )

                            st.code(
                                source["snippet"]
                            )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

                st.session_state.chat_history.append(
                    {
                        "question": user_query,
                        "answer": answer
                    }
                )
            except Exception as e:
                st.error(f"Execution Error: Connect to Ollama engine local server. Details {e}")
                
with col2:
    st.markdown("### System Ingestion & MLOps Control")

    pdfs = [
    f for f in os.listdir("data")
    if f.endswith(".pdf")
    ]

    st.metric(
        "Documents Loaded",
        len(pdfs)
    )

    # ADD HERE
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:

        file_path = f"data/{uploaded_file.name}"

        if not os.path.exists(file_path):

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(
                f"{uploaded_file.name} uploaded successfully."
            )

        else:
            st.info(
                f"{uploaded_file.name} already exists."
            )
                
    # Action button to trigger the ingest pipelines
    if st.button(" Parse & Index Local Documents", use_container_width=True):
        with st.spinner("Processing PDF directories via LangChain extraction..."):
            build_local_vector_db()
            st.success("Vector DB updated and mapped to persistent disk space!")

    st.markdown("---")
    """
    st.markdown("### Pipeline Quality Evaluation")

    # Action button to trigger the evaluation pipeline
    if st.button("☑ Run Ragas Statistical Evals", use_container_width=True):
        with st.spinner("Running automated validation loop over baseline metrics..."):
            try:
                scores = run_offline_ragas_evaluation()
                st.write("#### Core Quality Benchmark Scores:")
                st.json(scores)
            except Exception as e:
                st.warning(f"Evaluation suite ready structure validated. Requires live Ollama runtime to map scores: {e}")
                """