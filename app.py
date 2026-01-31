import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# 🔹 Load .env
load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    st.error("OPENAI_API_KEY not found in environment variables!")
client = OpenAI()

# 🔹 Paths
PROCESSED_DIR = "data/processed"
index_path = os.path.join(PROCESSED_DIR, "faiss_index.index")
chunks_meta_path = os.path.join(PROCESSED_DIR, "chunks_meta.json")

# 🔹 Load FAISS index and metadata
index = faiss.read_index(index_path)
with open(chunks_meta_path, "r", encoding="utf-8") as f:
    chunks_meta = json.load(f)

# 🔹 Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(MODEL_NAME)

# 🔹 Retrieval function with source filtering
def retrieve(query, top_k=5, selected_files=None):
    query_emb = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, top_k*2)
    results = []
    for i, idx in enumerate(indices[0]):
        chunk = chunks_meta[idx]
        if selected_files and chunk["filename"] not in selected_files:
            continue
        results.append({
            "rank": len(results)+1,
            "filename": chunk["filename"],
            "text": chunk["text"],
            "distance": float(distances[0][i])
        })
        if len(results) >= top_k:
            break
    return results

# 🔹 RAG function
def rag_query(query, top_k=5, selected_files=None):
    retrieved_chunks = retrieve(query, top_k, selected_files)
    context = "\n\n".join([c["text"] for c in retrieved_chunks])
    prompt = f"You have access to the following scientific publication fragments:\n\n{context}\n\nPlease provide a detailed answer to the question: {query}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in Raman spectroscopy and carbon nanotube nanostructures."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content, retrieved_chunks

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
st.title("🧪 RAG – Raman Nanotubes QA")

# 🔹 Sidebar (Portfolio info + PDF selection + live keywords)
st.sidebar.header("About this project")
st.sidebar.markdown("""
This demo showcases **RAG (Retrieval-Augmented Generation)** on scientific PDFs about **Raman spectroscopy of carbon nanotubes**.  
- Built with **Python, FAISS, Streamlit, OpenAI**  
- Allows selection of PDFs to focus retrieval  
- Highlights key terms in retrieved text
""")
all_files = list({chunk["filename"] for chunk in chunks_meta})
selected_files = st.sidebar.multiselect("Select PDFs for retrieval:", all_files, default=all_files[:5])

# 🔹 Extract live keywords based on selected PDFs
filtered_chunks = [c for c in chunks_meta if c["filename"] in selected_files]
top_keywords = extract_top_keywords(filtered_chunks, top_n=20)
highlight_keywords_selected = st.sidebar.multiselect("Highlight keywords:", top_keywords, default=top_keywords[:5])

# 🔹 Input
query = st.text_input("Your question:")
top_k = st.slider("Number of fragments to retrieve (top_k):", 1, 10, 5)

# 🔹 Ask button
if st.button("Ask question") and query:
    with st.spinner("Searching and generating answer..."):
        answer, retrieved_chunks = rag_query(query, top_k, selected_files)
    st.success("✅ Answer generated!")

    # Layout: two columns
    col1, col2 = st.columns([1,2])

    with col1:
        st.markdown("### 💬 Answer")
        st.text_area("", answer, height=300)

    with col2:
        st.markdown("### 📄 Top Retrieved Fragments")
        for chunk in retrieved_chunks:
            st.markdown(f"**Rank {chunk['rank']} | File:** {chunk['filename']} | Distance: {chunk['distance']:.3f}")
            st.write(highlight_keywords(chunk['text'][:500]+"...", highlight_keywords_selected))
            st.markdown("---")
