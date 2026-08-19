"""
FastAPI backend for the RAG system (Phase 1).

Reuses the same src/ modules (Retriever, rag_query) that power the Streamlit app —
no duplicated logic between the two interfaces.

Run with:
    uvicorn api:app --reload
"""
from dotenv import load_dotenv

load_dotenv()  # must run before importing src.generation, which creates an OpenAI client lazily but still needs the key available

from fastapi import FastAPI
from pydantic import BaseModel

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
    question: str
    top_k: int = config.DEFAULT_TOP_K


@app.get("/health")
def health_check():
    """Basic liveness check — confirms the server is up and responding."""
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest):
    """Runs the RAG pipeline: retrieves relevant chunks and generates an answer."""
    answer, retrieved_chunks = rag_query(retriever, request.question, top_k=request.top_k)
    return {"answer": answer, "sources": retrieved_chunks}