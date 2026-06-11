from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from src.llm_pipeline import execute_private_query
from src.vector_store import build_local_vector_db

app = FastAPI(title="Secure Enterprise RAG API")

class QueryRequest (BaseModel):
    question: str

@app.post("/api/v1/query")
async def secure_query_endpoint (request: QueryRequest):
    """Production asynchronous API endpoint for secure querying."""
    try:
        answer = execute_private_query(request.question)
        return {"status": "success", "data": {"answer": answer}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/api/v1/ingest")
async def trigger_ingestion (background_tasks: BackgroundTasks):
    """Triggers document processing in the background so the API doesn't freeze."""
    background_tasks.add_task(build_local_vector_db)
    return {"status": "accepted", "message": "Ingestion pipeline running in background."}