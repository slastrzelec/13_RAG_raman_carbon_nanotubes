import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # MUSI być przed importem src.generation, bo tam klient OpenAI tworzy się przy imporcie

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

from src.retrieval import Retriever
from src.generation import rag_query

# 🔹 Cache Retrievera - ciężkie zasoby (model, indeks) ładowane raz na sesję
@st.cache_resource
def load_retriever():
    return Retriever()

retriever = load_retriever()

# 🔹 Highlight keywords
def highlight_keywords(text, keywords):
    for kw in keywords:
        text = re.sub(f"({re.escape(kw)})", r"**\1**", text, flags=re.IGNORECASE)
    return text

# 🔹 Extract top keywords from corpus
def extract_top_keywords(chunks, top_n=20):
    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    X = vectorizer.fit_transform(corpus)
    scores = np.asarray(X.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    term_scores = list(zip(terms, scores))
    term_scores.sort(key=lambda x: x[1], reverse=True)
    top_terms = [t[0] for t in term_scores[:top_n]]
    return top_terms

# 🔹 Streamlit UI
st.set_page_config(page_title="RAG Raman Nanotubes", page_icon="🧪", layout="wide")

col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("# 🧪")
with col2:
    st.markdown("# RAG – Raman Nanotubes QA")

st.markdown("### 🔬 Semantic search engine for carbon nanotube research")
st.caption("v2 — hybrid search (dense + BM25), normalized cosine similarity")
st.divider()

# 🔹 Sidebar
st.sidebar.header("📚 About this project")
st.sidebar.markdown("""
**RAG (Retrieval-Augmented Generation)** on scientific PDFs about Raman spectroscopy of carbon nanotubes.

**Tech Stack:**
- Python, FAISS (IndexFlatIP, cosine similarity), Streamlit, OpenAI
- Hybrid search: dense embeddings + BM25
- Sentence-based chunking with deduplication
""")

all_files = sorted({chunk["filename"] for chunk in retriever.chunks_meta})
selected_files = st.sidebar.multiselect("📄 Select PDFs for retrieval:", all_files, default=all_files[:5])

use_hybrid = st.sidebar.toggle("🔀 Use hybrid search (dense + BM25)", value=True)

# 🔹 Live keywords based on selected PDFs
filtered_chunks = [c for c in retriever.chunks_meta if c["filename"] in selected_files]
top_keywords = extract_top_keywords(filtered_chunks, top_n=20) if filtered_chunks else []
highlight_keywords_selected = st.sidebar.multiselect("🔑 Highlight keywords:", top_keywords, default=top_keywords[:5])

st.sidebar.divider()
st.sidebar.metric("PDFs Selected", len(selected_files))
st.sidebar.metric("Chunks Available", len(filtered_chunks))

# 🔹 Input
DEFAULT_QUERY = "What is the D/G ratio in Raman spectroscopy and carbon nanotubes?"
query = st.text_input(
    "❓ Ask your question (English recommended — source documents are in English):",
    value=DEFAULT_QUERY,
    placeholder="e.g., What is RBM in carbon nanotubes?",
)
top_k = st.slider("📊 Fragments to retrieve:", 1, 10, 5)

if st.button("🔍 Ask question", type="primary") or query == DEFAULT_QUERY:
    with st.spinner("⏳ Searching and generating answer..."):
        answer, retrieved_chunks = rag_query(
            retriever, query, top_k=top_k, selected_files=selected_files
        )
    st.success("✅ Answer generated!")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 💬 Answer")
        st.text_area("", answer, height=300)

    with col2:
        st.markdown("### 📄 Top Retrieved Fragments")
        for chunk in retrieved_chunks:
            with st.container(border=True):
                st.markdown(f"**#{chunk['rank']}** • `{chunk['filename']}`")
                st.caption(f"Relevance score: {chunk['score']:.3f}")
                st.markdown(highlight_keywords(chunk['text'][:500] + "...", highlight_keywords_selected))