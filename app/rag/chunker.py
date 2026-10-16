import uuid
from typing import Any, Dict, List
from app.core.config import get_settings
from app.models.document import DocumentChunk

settings = get_settings()


class RecursiveCharacterChunker:
    """Splits text into semantically coherent chunks using recursive character delimiters."""

    def __init__(
        self,
        chunk_size: int = settings.RAG_CHUNK_SIZE,
        chunk_overlap: int = settings.RAG_CHUNK_OVERLAP,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        if not separators:
            # Base case: character split
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator = separators[0]
        splits = text.split(separator) if separator else list(text)
        next_separators = separators[1:]

        current_doc: List[str] = []
        total_len = 0

        for s in splits:
            item = s + (separator if separator else "")
            item_len = len(item)

            if item_len > self.chunk_size and next_separators:
                # Sub-split item that exceeds chunk size
                if current_doc:
                    joined = "".join(current_doc).strip()
                    if joined:
                        final_chunks.append(joined)
                    current_doc = []
                    total_len = 0
                sub_splits = self._split_text(item, next_separators)
                final_chunks.extend(sub_splits)
            elif total_len + item_len <= self.chunk_size:
                current_doc.append(item)
                total_len += item_len
            else:
                joined = "".join(current_doc).strip()
                if joined:
                    final_chunks.append(joined)
                # Overlap calculation
                overlap_text = "".join(current_doc)
                if self.chunk_overlap > 0 and len(overlap_text) > self.chunk_overlap:
                    current_doc = [overlap_text[-self.chunk_overlap:], item]
                    total_len = self.chunk_overlap + item_len
                else:
                    current_doc = [item]
                    total_len = item_len

        if current_doc:
            joined = "".join(current_doc).strip()
            if joined:
                final_chunks.append(joined)

        return final_chunks

    def chunk_document(
        self,
        document_id: str,
        text: str,
        metadata: Dict[str, Any] = None,
    ) -> List[DocumentChunk]:
        """Split a document's full text into a list of DocumentChunk models."""
        raw_chunks = self._split_text(text, self.separators)
        chunks: List[DocumentChunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue
            chunk_id = f"{document_id}#chunk_{idx}"
            chunk_metadata = (metadata or {}).copy()
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["character_count"] = len(chunk_text)

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                content=chunk_text,
                chunk_index=idx,
                metadata=chunk_metadata,
            )
            chunks.append(chunk)

        return chunks
