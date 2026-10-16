import pytest
from app.rag.chunker import RecursiveCharacterChunker


def test_chunker_basic_split():
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
    text = "RentCrew is an online platform for renting equipment. It offers seamless booking and support."
    chunks = chunker.chunk_document(document_id="doc_1", text=text, metadata={"filename": "test.txt"})

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.document_id == "doc_1"
        assert len(chunk.content) <= 60  # Within boundaries
        assert "filename" in chunk.metadata
        assert chunk.metadata["filename"] == "test.txt"


def test_chunker_empty_string():
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_document(document_id="doc_empty", text="")
    assert len(chunks) == 0


def test_chunker_indices_sequential():
    chunker = RecursiveCharacterChunker(chunk_size=30, chunk_overlap=5)
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = chunker.chunk_document(document_id="doc_seq", text=text)

    assert len(chunks) >= 2
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.chunk_id == f"doc_seq#chunk_{i}"
