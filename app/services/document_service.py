import uuid
from typing import Any, Dict, List, Optional
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.document import (
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentUploadResponse,
)
from app.rag.chunker import RecursiveCharacterChunker
from app.rag.embeddings import BaseEmbeddingProvider, get_embedding_provider
from app.rag.loader import DocumentLoader
from app.vectorstore.repository import VectorRepository

logger = get_logger(__name__)
settings = get_settings()


class DocumentService:
    """Manages document ingestion, chunking, vector embedding, and lifecycle operations."""

    def __init__(
        self,
        repository: Optional[VectorRepository] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        chunker: Optional[RecursiveCharacterChunker] = None,
    ):
        self.repository = repository or VectorRepository()
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.chunker = chunker or RecursiveCharacterChunker()

    async def ingest_bytes(
        self,
        filename: str,
        content: bytes,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentUploadResponse:
        """Process and index a document uploaded as raw bytes."""
        document_id = str(uuid.uuid4())
        logger.info("Ingesting document: %s (id: %s)", filename, document_id)

        # Step 1: Load and parse text
        text, doc_metadata = DocumentLoader.load_from_bytes(filename, content)
        if extra_metadata:
            doc_metadata.update(extra_metadata)

        # Step 2: Split text into semantic chunks
        chunks = self.chunker.chunk_document(
            document_id=document_id,
            text=text,
            metadata=doc_metadata,
        )
        if not chunks:
            logger.warning("No text extracted from document %s", filename)
            return DocumentUploadResponse(
                document_id=document_id,
                filename=filename,
                chunk_count=0,
                status="empty",
                message="Document contained no extractable text",
            )

        # Step 3: Compute vector embeddings in batch
        texts_to_embed = [c.content for c in chunks]
        embeddings = await self.embedding_provider.embed_documents(texts_to_embed)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        # Step 4: Index chunks in vector store
        indexed_count = await self.repository.upsert_chunks(chunks)
        logger.info("Successfully indexed %d chunks for document %s", indexed_count, filename)

        return DocumentUploadResponse(
            document_id=document_id,
            filename=filename,
            chunk_count=indexed_count,
            status="success",
            message=f"Indexed {indexed_count} chunks successfully",
        )

    async def ingest_file_path(self, file_path: str) -> DocumentUploadResponse:
        """Ingest a file from the local file system."""
        text, doc_metadata = DocumentLoader.load_from_path(file_path)
        filename = doc_metadata.get("filename", "unknown")
        document_id = str(uuid.uuid4())

        chunks = self.chunker.chunk_document(
            document_id=document_id,
            text=text,
            metadata=doc_metadata,
        )
        if not chunks:
            return DocumentUploadResponse(
                document_id=document_id,
                filename=filename,
                chunk_count=0,
                status="empty",
                message="File contained no extractable text",
            )

        embeddings = await self.embedding_provider.embed_documents([c.content for c in chunks])
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        indexed_count = await self.repository.upsert_chunks(chunks)
        return DocumentUploadResponse(
            document_id=document_id,
            filename=filename,
            chunk_count=indexed_count,
            status="success",
            message=f"Indexed {indexed_count} chunks successfully",
        )

    async def delete_document(self, document_id: str) -> DocumentDeleteResponse:
        """Delete an ingested document and its indexed vectors."""
        deleted = await self.repository.delete_document(document_id)
        return DocumentDeleteResponse(
            document_id=document_id,
            status="deleted",
            deleted_chunks=deleted,
        )

    async def list_documents(self) -> List[DocumentInfo]:
        """List all indexed documents."""
        return await self.repository.list_documents()
