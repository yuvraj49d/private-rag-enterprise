from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from src.llm_pipeline import execute_private_query
from src.vector_store import build_local_vector_db

app = FastAPI(
    title="Secure Enterprise RAG API",
    description="Offline Enterprise Retrieval-Augmented Generation System",
    version="1.0.0"
)

class QueryRequest (BaseModel):
    question: str

@app.post("/api/v1/query")
async def secure_query_endpoint (request: QueryRequest):
    """Production asynchronous API endpoint for secure querying."""
    try:
        result = execute_private_query(request.question)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/api/v1/ingest")
async def trigger_ingestion (background_tasks: BackgroundTasks):
    """Triggers document processing in the background so the API doesn't freeze."""
    background_tasks.add_task(build_local_vector_db)
    return {"status": "accepted", "message": "Ingestion pipeline running in background."}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": settings.LOCAL_LLM_MODEL,
        "embedding": settings.EMBEDDING_MODEL_NAME
    }

@app.get("/metrics")
async def metrics():
    return {
        "documents": len([
            f for f in os.listdir("data")
            if f.endswith(".pdf")
        ])
    }