# Enterprise Private RAG Architecture

User
│
├── Streamlit Dashboard
│
├── FastAPI REST API
│
▼
Hybrid Retriever
│
├── Chroma Vector Search
│
└── BM25 Keyword Search
│
▼
CrossEncoder Reranker
│
▼
Context Builder
│
▼
Ollama (Qwen2.5)
│
▼
Response Generator
│
▼
Source Attribution

Data Pipeline
│
▼
PDF Upload
│
▼
Document Loader
│
▼
Text Chunking
│
▼
Embedding Generation
│
▼
Chroma Persistence

Evaluation Layer
│
▼
RAGAS
│
├── Answer Relevancy
├── Faithfulness
└── Context Precision
