"""
Testy jednostkowe dla src/ingestion.py.
Nie wymagają prawdziwych PDF-ów ani kluczy API — testują czystą logikę.

Uruchomienie: pytest tests/test_ingestion.py -v
"""
import pytest

from src.ingestion import split_into_sentences, chunk_document, deduplicate_documents


class TestSplitIntoSentences:
    def test_splits_basic_sentences(self):
        text = "This is sentence one. This is sentence two. Is this sentence three?"
        sentences = split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "This is sentence one."

    def test_handles_extra_whitespace(self):
        text = "Sentence one.    Sentence   two."
        sentences = split_into_sentences(text)
        assert len(sentences) == 2

    def test_empty_text_returns_empty_list(self):
        assert split_into_sentences("") == []


class TestChunkDocument:
    def test_produces_at_least_one_chunk(self):
        text = "First sentence here. Second sentence here. Third one too."
        chunks = chunk_document("test.pdf", text, chunk_size_tokens=5, overlap_tokens=2)
        assert len(chunks) >= 1
        assert all(c["filename"] == "test.pdf" for c in chunks)

    def test_chunks_have_overlap(self):
        text = " ".join([f"Sentence number {i} here." for i in range(20)])
        chunks = chunk_document("test.pdf", text, chunk_size_tokens=10, overlap_tokens=5)
        assert len(chunks) > 1
        # sprawdzamy, że kolejne chunki dzielą jakiś wspólny fragment (overlap działa)
        first_words = set(chunks[0]["text"].split())
        second_words = set(chunks[1]["text"].split())
        assert len(first_words & second_words) > 0

    def test_empty_document_produces_no_chunks(self):
        chunks = chunk_document("empty.pdf", "", chunk_size_tokens=100, overlap_tokens=20)
        assert chunks == []


class TestDeduplicateDocuments:
    def test_removes_exact_duplicate_content(self):
        docs = [
            {"filename": "a.pdf", "text": "Some scientific content about nanotubes."},
            {"filename": "b_copy.pdf", "text": "Some scientific content about nanotubes."},
        ]
        unique = deduplicate_documents(docs)
        assert len(unique) == 1

    def test_keeps_distinct_content(self):
        docs = [
            {"filename": "a.pdf", "text": "Content about Raman spectroscopy."},
            {"filename": "b.pdf", "text": "Content about carbon nanotube synthesis."},
        ]
        unique = deduplicate_documents(docs)
        assert len(unique) == 2

    def test_ignores_whitespace_differences_when_deduplicating(self):
        docs = [
            {"filename": "a.pdf", "text": "Some   content   here."},
            {"filename": "b.pdf", "text": "Some content here."},
        ]
        unique = deduplicate_documents(docs)
        assert len(unique) == 1
