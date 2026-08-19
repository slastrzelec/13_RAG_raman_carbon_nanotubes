"""
Ingestion pipeline: PDF -> tekst -> deduplikacja -> chunking po zdaniach ->
embeddingi -> znormalizowany indeks FAISS (IndexFlatIP = cosine similarity).

Faza 0 — poprawki względem oryginalnego notebooka 01_load_pdfs.ipynb:
1. IndexFlatIP + L2-normalizacja wektorów zamiast IndexFlatL2 (poprawny cosine similarity)
2. Chunking po zdaniach/akapitach zamiast sztywnego cięcia po znakach
3. Deduplikacja treści źródłowych przed budową indeksu (wykryto zdublowane PDF-y,
   np. "Jorio_2003..." i "R11 Jorio_2003..." to ten sam artykuł)

Uruchomienie:
    python -m src.ingestion
"""
import os
import re
import json
import hashlib

import fitz  # pymupdf
import numpy as np
import faiss
from tqdm import tqdm
# SentenceTransformer importowany leniwie w build_faiss_index() —
# testy logiki tekstowej (chunking, dedup) nie powinny wymagać ciężkich zależności ML

from src import config
from src.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_pdfs(raw_dir: str) -> list[dict]:
    """Wyciąga tekst ze wszystkich PDF-ów w raw_dir."""
    pdf_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".pdf")]
    logger.info(f"Znaleziono {len(pdf_files)} plików PDF w {raw_dir}", extra={"num_pdfs": len(pdf_files), "raw_dir": raw_dir})

    documents = []
    for pdf_file in tqdm(pdf_files, desc="Ekstrakcja tekstu z PDF"):
        pdf_path = os.path.join(raw_dir, pdf_file)
        doc = fitz.open(pdf_path)
        pages_text = [doc[p].get_text() for p in range(len(doc))]
        full_text = "\n".join(pages_text)
        documents.append({
            "filename": pdf_file,
            "num_pages": len(doc),
            "text": full_text,
        })
        doc.close()
    return documents


def deduplicate_documents(documents: list[dict]) -> list[dict]:
    """
    Usuwa duplikaty treści (nie tylko duplikaty nazw plików).
    W danych źródłowych wykryto artykuły zapisane pod dwiema różnymi nazwami
    (np. ten sam paper jako 'Jorio_2003...' i 'R11 Jorio_2003...') —
    identyczna treść dawała identyczne, zdublowane wyniki w retrievalu.
    """
    seen_hashes = {}
    unique_docs = []
    duplicates_found = []

    for doc in documents:
        # Hash znormalizowanej treści (bez białych znaków na brzegach, lowercase)
        normalized = re.sub(r"\s+", " ", doc["text"]).strip().lower()
        content_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()

        if content_hash in seen_hashes:
            duplicates_found.append((doc["filename"], seen_hashes[content_hash]))
            continue

        seen_hashes[content_hash] = doc["filename"]
        unique_docs.append(doc)

    if duplicates_found:
        logger.warning(
            f"Wykryto {len(duplicates_found)} duplikatów treści",
            extra={"num_duplicates": len(duplicates_found), "duplicates": duplicates_found},
        )
        for dup, original in duplicates_found:
            logger.warning(f"  '{dup}' jest duplikatem '{original}' — pominięto", extra={"duplicate_file": dup, "original_file": original})
    else:
        logger.info("Brak duplikatów treści.")

    logger.info(
        f"Dokumenty po deduplikacji: {len(unique_docs)} (z {len(documents)})",
        extra={"num_unique": len(unique_docs), "num_total": len(documents)},
    )
    return unique_docs


def split_into_sentences(text: str) -> list[str]:
    """
    Prosty, zależny-tylko-od-regex podział na zdania (bez dodatkowych zależności typu nltk).
    Wystarczający dla tekstu naukowego w języku angielskim.
    """
    # Scalanie białych znaków (PDF-y mają dużo dziwnych łamań linii)
    text = re.sub(r"\s+", " ", text).strip()
    # Podział po kropce/wykrzykniku/pytajniku + spacja + wielka litera (heurystyka)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(filename: str, text: str, chunk_size_tokens: int, overlap_tokens: int) -> list[dict]:
    """
    Chunking po zdaniach (nie po surowych znakach jak w oryginalnym notebooku).
    Zdania są grupowane w chunki o docelowej długości ~chunk_size_tokens słów,
    z zachowaniem overlapu między kolejnymi chunkami, żeby nie tracić kontekstu na granicach.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_sentences: list[str] = []
    current_word_count = 0

    def flush_chunk():
        if not current_sentences:
            return None
        chunk_text = " ".join(current_sentences)
        return chunk_text

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_word_count = len(sentence.split())

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

        if current_word_count >= chunk_size_tokens:
            chunk_text = flush_chunk()
            chunks.append({
                "filename": filename,
                "text": chunk_text,
            })

            # Overlap: cofamy się o tyle zdań, ile odpowiada ~overlap_tokens słów
            overlap_words = 0
            overlap_sentences = []
            for s in reversed(current_sentences):
                overlap_words += len(s.split())
                overlap_sentences.insert(0, s)
                if overlap_words >= overlap_tokens:
                    break

            current_sentences = overlap_sentences
            current_word_count = overlap_words

        i += 1

    # Ostatni, niepełny chunk
    if current_sentences:
        chunk_text = flush_chunk()
        if chunk_text:
            chunks.append({
                "filename": filename,
                "text": chunk_text,
            })

    return chunks


def build_chunks(documents: list[dict]) -> list[dict]:
    all_chunks = []
    for doc in tqdm(documents, desc="Chunking po zdaniach"):
        doc_chunks = chunk_document(
            filename=doc["filename"],
            text=doc["text"],
            chunk_size_tokens=config.CHUNK_SIZE_TOKENS,
            overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
        )
        all_chunks.extend(doc_chunks)
    logger.info(
        f"Utworzono {len(all_chunks)} chunków (docelowo ~{config.CHUNK_SIZE_TOKENS} słów każdy)",
        extra={"num_chunks": len(all_chunks), "target_chunk_size_tokens": config.CHUNK_SIZE_TOKENS},
    )
    return all_chunks


def build_faiss_index(chunks: list[dict]) -> tuple[faiss.Index, np.ndarray]:
    """
    Faza 0, kluczowa poprawka:
    IndexFlatIP (inner product) + L2-normalizacja wektorów = poprawny cosine similarity.
    Oryginalny kod używał IndexFlatL2 na nienormalizowanych wektorach, co daje
    matematycznie inną (gorszą, wrażliwą na długość wektora) miarę podobieństwa.
    """
    from sentence_transformers import SentenceTransformer  # leniwy import (ciężka zależność)
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    texts = [c["text"] for c in chunks]

    logger.info(f"Generowanie embeddingów dla {len(texts)} chunków...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings).astype("float32")

    # L2 normalizacja — po tym kroku inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    index.add(embeddings)
    logger.info(
        f"Zbudowano IndexFlatIP z {index.ntotal} wektorami (znormalizowane, cosine similarity)",
        extra={"num_vectors": index.ntotal, "embedding_dim": config.EMBEDDING_DIM},
    )

    return index, embeddings


def run_ingestion_pipeline():
    """Pełny pipeline: PDF -> dedup -> chunking -> embeddingi -> zapis na dysk."""
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    documents = extract_text_from_pdfs(config.RAW_DIR)
    documents = deduplicate_documents(documents)
    chunks = build_chunks(documents)
    index, _ = build_faiss_index(chunks)

    faiss.write_index(index, config.INDEX_PATH)
    with open(config.CHUNKS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Gotowe. Indeks: {config.INDEX_PATH}", extra={"index_path": config.INDEX_PATH})
    logger.info(f"Metadata chunków: {config.CHUNKS_META_PATH}", extra={"chunks_meta_path": config.CHUNKS_META_PATH})


if __name__ == "__main__":
    run_ingestion_pipeline()
