# RAG – Raman Nanotubes QA

A Retrieval-Augmented Generation (RAG) system for semantic search and Q&A over scientific
publications on Raman spectroscopy of carbon nanotubes. Started as a simple Streamlit demo,
now being rebuilt into a production-grade RAG API (v2-production branch).

**Live demo:** [carbon-nanotubesrag.streamlit.app](https://carbon-nanotubesrag.streamlit.app/)

## Overview

- Semantic + keyword hybrid search over 25 scientific PDFs on Raman spectroscopy of carbon nanotubes
- LLM-generated answers grounded in retrieved context, with source citations
- Evaluated for factual faithfulness using [RAGAs](https://github.com/explodinggradients/ragas)

## Tech Stack

- **Language / core:** Python
- **Retrieval:** FAISS (`IndexFlatIP`, normalized embeddings → cosine similarity), `sentence-transformers` (`all-MiniLM-L6-v2`), BM25 (hybrid search)
- **Generation:** OpenAI API (`gpt-4o-mini`)
- **UI:** Streamlit
- **Evaluation:** RAGAs (faithfulness)
- **Testing:** pytest

## Architecture

```
app.py                  → Streamlit UI (thin layer, imports from src/)
api.py                   → FastAPI backend (thin layer, imports from src/)
src/
  config.py              → paths, model names, chunking/retrieval parameters
  ingestion.py            → PDF → text → dedup → sentence-based chunking → FAISS index
  retrieval.py             → dense (FAISS/cosine) + BM25 hybrid search
  generation.py             → prompt construction + LLM call
  eval_dataset.py            → fixed set of evaluation questions
  evaluation.py               → RAGAs evaluation script
tests/
  test_ingestion.py            → unit tests for chunking, deduplication
  test_api.py                   → unit tests for FastAPI endpoints (mocked, no real API calls)
data/
  raw/                          → source PDFs (not tracked in git)
  processed/                     → FAISS index + chunk metadata
  evaluation/                     → RAGAs results (JSON)
```

Both `app.py` and `api.py` are thin interface layers over the same `src/` modules —
no duplicated retrieval or generation logic between the Streamlit UI and the REST API.

## Phase 0 — Retrieval Quality & Engineering Foundations

The original prototype used `IndexFlatL2` on non-normalized embeddings and naive
character-based chunking. Phase 0 focused on fixing retrieval correctness and adding
the engineering practices (tests, evaluation) that were identified as gaps in
technical interview feedback.

**What changed:**

- **Correct similarity metric** — switched to `IndexFlatIP` with L2-normalized vectors,
  giving mathematically correct cosine similarity instead of raw Euclidean distance on
  unnormalized vectors.
- **Sentence-based chunking** — replaced fixed 500-character cuts with sentence-aware
  chunking (~120 words per chunk, with overlap), avoiding mid-sentence truncation.
- **Source deduplication** — added content-hash-based deduplication of source PDFs.
  This caught real duplicates in the dataset (the same paper saved under two different
  filenames), which had previously caused identical, redundant results to appear
  in the top of every search.
- **Hybrid search** — combined dense (embedding) retrieval with BM25 keyword search,
  to catch matches that pure semantic search can miss (e.g. exact technical terms,
  abbreviations like "RBM", "SWNT").
- **Modular architecture** — split the original single-file Streamlit app into
  `ingestion` / `retrieval` / `generation` modules, independently testable and reusable
  (e.g. by a future FastAPI layer).
- **Unit tests** — 9 tests covering chunking and deduplication logic (`pytest tests/`).

## Phase 1 — REST API (FastAPI)

Added a FastAPI backend alongside the existing Streamlit app, so the RAG pipeline
can be consumed by any client (curl, another service, a future integration) — not
just through the Streamlit UI. Both interfaces share the exact same `src/` modules.

**What was added:**

- **Three endpoints:**
  - `GET /health` — liveness check
  - `POST /query` — runs the RAG pipeline (retrieval + generation), returns the
    answer and its sources
  - `GET /documents` — lists all indexed source PDFs
- **Request validation** (Pydantic) — questions must be non-empty and non-whitespace;
  `top_k` is bounded to a sane range (1–20). Invalid requests are rejected with a
  clear `422` response before any retrieval or LLM call happens.
- **Error handling** — failures in the retrieval/generation pipeline (e.g. an OpenAI
  API timeout or rate limit) are caught and returned as a `502` with a readable
  message, instead of a raw stack trace.
- **Automatic interactive docs** — available at `/docs` (Swagger UI), generated
  directly from the Pydantic models and endpoint definitions.
- **Unit tests** (`tests/test_api.py`) — 10 tests covering all three endpoints,
  validation edge cases (empty/whitespace questions, out-of-range `top_k`, missing
  fields), and pipeline failure handling. These mock `rag_query()` and the
  `Retriever`, so they run in seconds without a real OpenAI key or a built index —
  they test the API layer itself, not RAG answer quality (that's covered separately
  by the RAGAs evaluation in Phase 0).

Run the API locally:
```bash
uvicorn api:app --reload
```
Then open `http://127.0.0.1:8000/docs` for the interactive documentation.

## Evaluation (RAGAs)

The system was evaluated on a fixed set of 14 domain questions (see
`src/eval_dataset.py`), covering core Raman spectroscopy concepts (D band, G band,
D/G ratio, RBM, chirality, defect characterization), measurement methodology, and one
deliberately out-of-scope question (as a hallucination sanity check).

**Metric: faithfulness** — does the generated answer stick to facts present in the
retrieved context, without introducing unsupported claims?

| Metric | Result |
|---|---|
| Mean faithfulness | **~0.80–0.82** (14 questions) |

**Observations:**

- The out-of-scope question ("What is the melting point of carbon nanotubes?")
  scored **1.0** — the system correctly stated the context didn't contain this
  information instead of fabricating an answer.
- Lower-scoring answers were manually inspected by reviewing the full retrieved
  context and generated answer for each question (saved alongside the faithfulness
  score in `data/evaluation/ragas_results.json`). In most low-scoring cases, retrieval
  was accurate — the retrieved chunks did contain the relevant facts — but the
  generated answer additionally drew on general domain knowledge not explicitly
  present in the retrieved passages (e.g. correct textbook facts about the G-band's
  relation to the E2g phonon mode). This is a case of the model enriching answers
  with accurate general knowledge, not hallucination, but it is worth noting as a
  known characteristic of the current prompt design.
- `context_precision` was intentionally left out of this evaluation round, since it
  requires reference (ground-truth) answers that this question set does not include.

Run the evaluation yourself:
```bash
python -m src.evaluation
```
Note: this makes real OpenAI API calls (both for answer generation and RAGAs'
LLM-as-judge scoring) and is not part of the automated test suite.

## Running Locally

```bash
# install dependencies
pip install -r requirements.txt        # or requirements-dev.txt for tests/evaluation

# build the index from PDFs in data/raw/
python -m src.ingestion

# run the Streamlit app
streamlit run app.py

# run the REST API
uvicorn api:app --reload

# run tests
pytest tests/ -v
```

## Roadmap

- [x] Phase 0 — retrieval correctness, hybrid search, testing, evaluation
- [x] Phase 1 — FastAPI backend, request/response validation, error handling, API tests
- [ ] Phase 2 — structured logging
- [ ] Phase 3 — Docker + docker-compose
- [ ] Phase 4 — CI/CD (GitHub Actions)
- [ ] Phase 5 — deployment (Render/Railway)
- [ ] Phase 6 — architecture docs, expanded evaluation (context precision with reference answers)# RAG – Raman Nanotubes QA

A Retrieval-Augmented Generation (RAG) system for semantic search and Q&A over scientific
publications on Raman spectroscopy of carbon nanotubes. Started as a simple Streamlit demo,
now being rebuilt into a production-grade RAG API (v2-production branch).

**Live demo:** [carbon-nanotubesrag.streamlit.app](https://carbon-nanotubesrag.streamlit.app/)

## Overview

- Semantic + keyword hybrid search over 25 scientific PDFs on Raman spectroscopy of carbon nanotubes
- LLM-generated answers grounded in retrieved context, with source citations
- Evaluated for factual faithfulness using [RAGAs](https://github.com/explodinggradients/ragas)

## Tech Stack

- **Language / core:** Python
- **Retrieval:** FAISS (`IndexFlatIP`, normalized embeddings → cosine similarity), `sentence-transformers` (`all-MiniLM-L6-v2`), BM25 (hybrid search)
- **Generation:** OpenAI API (`gpt-4o-mini`)
- **UI:** Streamlit
- **Evaluation:** RAGAs (faithfulness)
- **Testing:** pytest

## Architecture

```
app.py                  → Streamlit UI (thin layer, imports from src/)
api.py                   → FastAPI backend (thin layer, imports from src/)
src/
  config.py              → paths, model names, chunking/retrieval parameters
  ingestion.py            → PDF → text → dedup → sentence-based chunking → FAISS index
  retrieval.py             → dense (FAISS/cosine) + BM25 hybrid search
  generation.py             → prompt construction + LLM call
  eval_dataset.py            → fixed set of evaluation questions
  evaluation.py               → RAGAs evaluation script
tests/
  test_ingestion.py            → unit tests for chunking, deduplication
  test_api.py                   → unit tests for FastAPI endpoints (mocked, no real API calls)
data/
  raw/                          → source PDFs (not tracked in git)
  processed/                     → FAISS index + chunk metadata
  evaluation/                     → RAGAs results (JSON)
```

Both `app.py` and `api.py` are thin interface layers over the same `src/` modules —
no duplicated retrieval or generation logic between the Streamlit UI and the REST API.

## Phase 0 — Retrieval Quality & Engineering Foundations

The original prototype used `IndexFlatL2` on non-normalized embeddings and naive
character-based chunking. Phase 0 focused on fixing retrieval correctness and adding
the engineering practices (tests, evaluation) that were identified as gaps in
technical interview feedback.

**What changed:**

- **Correct similarity metric** — switched to `IndexFlatIP` with L2-normalized vectors,
  giving mathematically correct cosine similarity instead of raw Euclidean distance on
  unnormalized vectors.
- **Sentence-based chunking** — replaced fixed 500-character cuts with sentence-aware
  chunking (~120 words per chunk, with overlap), avoiding mid-sentence truncation.
- **Source deduplication** — added content-hash-based deduplication of source PDFs.
  This caught real duplicates in the dataset (the same paper saved under two different
  filenames), which had previously caused identical, redundant results to appear
  in the top of every search.
- **Hybrid search** — combined dense (embedding) retrieval with BM25 keyword search,
  to catch matches that pure semantic search can miss (e.g. exact technical terms,
  abbreviations like "RBM", "SWNT").
- **Modular architecture** — split the original single-file Streamlit app into
  `ingestion` / `retrieval` / `generation` modules, independently testable and reusable
  (e.g. by a future FastAPI layer).
- **Unit tests** — 9 tests covering chunking and deduplication logic (`pytest tests/`).

## Phase 1 — REST API (FastAPI)

Added a FastAPI backend alongside the existing Streamlit app, so the RAG pipeline
can be consumed by any client (curl, another service, a future integration) — not
just through the Streamlit UI. Both interfaces share the exact same `src/` modules.

**What was added:**

- **Three endpoints:**
  - `GET /health` — liveness check
  - `POST /query` — runs the RAG pipeline (retrieval + generation), returns the
    answer and its sources
  - `GET /documents` — lists all indexed source PDFs
- **Request validation** (Pydantic) — questions must be non-empty and non-whitespace;
  `top_k` is bounded to a sane range (1–20). Invalid requests are rejected with a
  clear `422` response before any retrieval or LLM call happens.
- **Error handling** — failures in the retrieval/generation pipeline (e.g. an OpenAI
  API timeout or rate limit) are caught and returned as a `502` with a readable
  message, instead of a raw stack trace.
- **Automatic interactive docs** — available at `/docs` (Swagger UI), generated
  directly from the Pydantic models and endpoint definitions.
- **Unit tests** (`tests/test_api.py`) — 10 tests covering all three endpoints,
  validation edge cases (empty/whitespace questions, out-of-range `top_k`, missing
  fields), and pipeline failure handling. These mock `rag_query()` and the
  `Retriever`, so they run in seconds without a real OpenAI key or a built index —
  they test the API layer itself, not RAG answer quality (that's covered separately
  by the RAGAs evaluation in Phase 0).

Run the API locally:
```bash
uvicorn api:app --reload
```
Then open `http://127.0.0.1:8000/docs` for the interactive documentation.

## Evaluation (RAGAs)

The system was evaluated on a fixed set of 14 domain questions (see
`src/eval_dataset.py`), covering core Raman spectroscopy concepts (D band, G band,
D/G ratio, RBM, chirality, defect characterization), measurement methodology, and one
deliberately out-of-scope question (as a hallucination sanity check).

**Metric: faithfulness** — does the generated answer stick to facts present in the
retrieved context, without introducing unsupported claims?

| Metric | Result |
|---|---|
| Mean faithfulness | **~0.80–0.82** (14 questions) |

**Observations:**

- The out-of-scope question ("What is the melting point of carbon nanotubes?")
  scored **1.0** — the system correctly stated the context didn't contain this
  information instead of fabricating an answer.
- Lower-scoring answers were manually inspected by reviewing the full retrieved
  context and generated answer for each question (saved alongside the faithfulness
  score in `data/evaluation/ragas_results.json`). In most low-scoring cases, retrieval
  was accurate — the retrieved chunks did contain the relevant facts — but the
  generated answer additionally drew on general domain knowledge not explicitly
  present in the retrieved passages (e.g. correct textbook facts about the G-band's
  relation to the E2g phonon mode). This is a case of the model enriching answers
  with accurate general knowledge, not hallucination, but it is worth noting as a
  known characteristic of the current prompt design.
- `context_precision` was intentionally left out of this evaluation round, since it
  requires reference (ground-truth) answers that this question set does not include.

Run the evaluation yourself:
```bash
python -m src.evaluation
```
Note: this makes real OpenAI API calls (both for answer generation and RAGAs'
LLM-as-judge scoring) and is not part of the automated test suite.

## Running Locally

```bash
# install dependencies
pip install -r requirements.txt        # or requirements-dev.txt for tests/evaluation

# build the index from PDFs in data/raw/
python -m src.ingestion

# run the Streamlit app
streamlit run app.py

# run the REST API
uvicorn api:app --reload

# run tests
pytest tests/ -v
```

## Roadmap

- [x] Phase 0 — retrieval correctness, hybrid search, testing, evaluation
- [x] Phase 1 — FastAPI backend, request/response validation, error handling, API tests
- [ ] Phase 2 — structured logging
- [ ] Phase 3 — Docker + docker-compose
- [ ] Phase 4 — CI/CD (GitHub Actions)
- [ ] Phase 5 — deployment (Render/Railway)
- [ ] Phase 6 — architecture docs, expanded evaluation (context precision with reference answers)