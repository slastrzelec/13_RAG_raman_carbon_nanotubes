"""
Moduł retrievalu: dense search (FAISS, cosine similarity) + hybrid z BM25.

Faza 0 — poprawki:
1. Zapytania też są L2-normalizowane przed wyszukiwaniem (spójność z indeksem IndexFlatIP)
2. Hybrid search: łączy dense retrieval z BM25 (rank_bm25), żeby łapać dopasowania
   leksykalne/keyword, których czysty embedding może nie wychwycić (np. rzadkie
   nazwy własne, wzory, skróty typu "RBM", "SWNT")
"""
import json
import logging

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src import config

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, index_path: str = config.INDEX_PATH, chunks_meta_path: str = config.CHUNKS_META_PATH):
        self.index: faiss.Index = faiss.read_index(index_path)
        with open(chunks_meta_path, "r", encoding="utf-8") as f:
            self.chunks_meta: list[dict] = json.load(f)

        self.embed_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

        # BM25 nad tymi samymi chunkami — indeks leksykalny obok wektorowego
        tokenized_corpus = [c["text"].lower().split() for c in self.chunks_meta]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _dense_search(self, query: str, top_k: int) -> dict[int, float]:
        """Zwraca {chunk_index: znormalizowany_similarity_score}."""
        query_emb = self.embed_model.encode([query]).astype("float32")
        faiss.normalize_L2(query_emb)  # spójne z normalizacją w ingestion.py
        scores, indices = self.index.search(query_emb, top_k)

        results = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results[int(idx)] = float(score)  # już w [-1, 1] dzięki cosine similarity
        return results

    def _bm25_search(self, query: str, top_k: int) -> dict[int, float]:
        """Zwraca {chunk_index: znormalizowany_bm25_score}."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]
        max_score = scores[top_indices[0]] if len(top_indices) > 0 and scores[top_indices[0]] > 0 else 1.0

        return {int(idx): float(scores[idx] / max_score) for idx in top_indices}

    def retrieve(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        selected_files: list[str] | None = None,
        use_hybrid: bool = True,
        alpha: float = config.HYBRID_ALPHA,
    ) -> list[dict]:
        """
        Zwraca top_k najbardziej trafnych chunków.

        alpha kontroluje wagę dense vs BM25:
            final_score = alpha * dense_score + (1 - alpha) * bm25_score
        alpha=1.0 -> czysty dense, alpha=0.0 -> czysty BM25.
        """
        candidate_pool = top_k * 4  # szukamy szerzej, potem filtrujemy i łączymy

        dense_scores = self._dense_search(query, candidate_pool)

        if use_hybrid:
            bm25_scores = self._bm25_search(query, candidate_pool)
            all_indices = set(dense_scores) | set(bm25_scores)
            combined_scores = {
                idx: alpha * dense_scores.get(idx, 0.0) + (1 - alpha) * bm25_scores.get(idx, 0.0)
                for idx in all_indices
            }
        else:
            combined_scores = dense_scores

        ranked_indices = sorted(combined_scores.keys(), key=lambda i: combined_scores[i], reverse=True)

        results = []
        for idx in ranked_indices:
            chunk = self.chunks_meta[idx]
            if selected_files and chunk["filename"] not in selected_files:
                continue
            results.append({
                "rank": len(results) + 1,
                "filename": chunk["filename"],
                "text": chunk["text"],
                "score": round(combined_scores[idx], 4),
            })
            if len(results) >= top_k:
                break

        return results
