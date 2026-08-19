"""
FastAPI backend for the RAG system (Phase 1).

Reuses the same src/ modules (Retriever, rag_query) that power the Streamlit app —
no duplicated logic between the two interfaces.

Run with:
    uvicorn api:app --reload
"""
from dotenv import load_dotenv

load_dotenv()  # must run before importing src.generation, which creates an OpenAI client lazily but still needs the key available

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.retrieval import Retriever
from src.generation import rag_query
from src import config

app = FastAPI(
    title="RAG Raman Nanotubes API",
    description="REST API for semantic search and Q&A over scientific papers on Raman spectroscopy of carbon nanotubes.",
    version="0.1.0",
)

# Loaded once at startup, reused across requests — avoids reloading the embedding
# model and FAISS index on every call.
retriever = Retriever()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask, must not be empty.")
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=20, description="Number of chunks to retrieve (1-20).")


@app.get("/health")
def health_check():
    """Basic liveness check — confirms the server is up and responding."""
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest):
    """Runs the RAG pipeline: retrieves relevant chunks and generates an answer."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty or whitespace-only.")

    try:
        answer, retrieved_chunks = rag_query(retriever, question, top_k=request.top_k)
    except Exception as e:
        # Covers OpenAI API errors (rate limits, timeouts, auth issues) and any
        # unexpected failure in the retrieval/generation pipeline — the caller
        # gets a clear 502 instead of a raw stack trace.
        raise HTTPException(status_code=502, detail=f"Failed to generate an answer: {e}")

    return {"answer": answer, "sources": retrieved_chunks}


@app.get("/documents")
def list_documents():
    """Lists all unique source PDF filenames currently indexed."""
    filenames = sorted({chunk["filename"] for chunk in retriever.chunks_meta})
    return {"count": len(filenames), "documents": filenames}