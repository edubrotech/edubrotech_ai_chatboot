import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.chat import SourceDocument
from app.models.document import DocumentChunk, DocumentInfo
from app.vectorstore.qdrant import get_async_qdrant_client, init_qdrant_collection

logger = get_logger(__name__)
settings = get_settings()


class VectorRepository:
    """Repository handling vector upsert, search, and document metadata indexing in Qdrant."""

    def __init__(self, client: Optional[AsyncQdrantClient] = None):
        self.client = client or get_async_qdrant_client()
        self.collection_name = settings.QDRANT_COLLECTION
        # In-memory fallback storage if Qdrant is offline
        self._memory_chunks: Dict[str, DocumentChunk] = {}

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two float vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    async def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Upsert a list of document chunks and their embeddings into the vector store."""
        if not chunks:
            return 0

        try:
            await init_qdrant_collection()
            points = []
            for chunk in chunks:
                if not chunk.embedding:
                    continue
                # Generate a stable deterministic UUID from chunk_id for Qdrant point IDs
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                    "filename": chunk.metadata.get("filename", "unknown"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=chunk.embedding,
                        payload=payload,
                    )
                )

            if points:
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                logger.info("Upserted %d points to Qdrant collection '%s'", len(points), self.collection_name)
            return len(points)

        except Exception as e:
            logger.warning("Qdrant unavailable (%s), storing %d chunks in memory fallback", str(e), len(chunks))
            for chunk in chunks:
                self._memory_chunks[chunk.chunk_id] = chunk
            return len(chunks)

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        score_threshold: float = 0.0,
    ) -> List[SourceDocument]:
        """Perform similarity search for a query embedding vector."""
        try:
            search_result = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
            )

            documents: List[SourceDocument] = []
            for point in search_result:
                payload = point.payload or {}
                documents.append(
                    SourceDocument(
                        document_id=payload.get("document_id", "unknown"),
                        chunk_id=payload.get("chunk_id", str(point.id)),
                        content=payload.get("content", ""),
                        score=float(point.score),
                        metadata=payload.get("metadata", {}),
                    )
                )
            return documents

        except Exception as e:
            logger.warning("Qdrant search failed (%s), searching memory fallback", str(e))
            # Fallback search across in-memory chunks
            scored: List[SourceDocument] = []
            for chunk in self._memory_chunks.values():
                if chunk.embedding:
                    score = self._cosine_similarity(query_vector, chunk.embedding)
                    if score >= score_threshold:
                        scored.append(
                            SourceDocument(
                                document_id=chunk.document_id,
                                chunk_id=chunk.chunk_id,
                                content=chunk.content,
                                score=score,
                                metadata=chunk.metadata,
                            )
                        )
            scored.sort(key=lambda x: x.score, reverse=True)
            return scored[:top_k]

    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document_id."""
        deleted_count = 0
        try:
            delete_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=document_id),
                    )
                ]
            )
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(filter=delete_filter),
            )
            logger.info("Deleted chunks for document_id '%s' from Qdrant", document_id)
            deleted_count = 1
        except Exception as e:
            logger.warning("Qdrant delete error (%s), cleaning fallback memory", str(e))

        # Also purge from memory fallback
        to_del = [k for k, v in self._memory_chunks.items() if v.document_id == document_id]
        for k in to_del:
            del self._memory_chunks[k]
        deleted_count = max(deleted_count, len(to_del))

        return deleted_count

    async def list_documents(self) -> List[DocumentInfo]:
        """List distinct ingested documents with metadata."""
        docs_map: Dict[str, Dict[str, Any]] = {}

        try:
            # Scroll points to collect document IDs and counts
            scroll_result, _ = await self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            for point in scroll_result:
                payload = point.payload or {}
                doc_id = payload.get("document_id")
                if not doc_id:
                    continue
                if doc_id not in docs_map:
                    docs_map[doc_id] = {
                        "document_id": doc_id,
                        "filename": payload.get("filename", "unknown"),
                        "chunk_count": 0,
                        "metadata": payload.get("metadata", {}),
                        "created_at": datetime.now(timezone.utc),
                    }
                docs_map[doc_id]["chunk_count"] += 1
        except Exception as e:
            logger.debug("Qdrant scroll unavailable (%s), inspecting memory fallback", str(e))

        # Combine with memory fallback
        for chunk in self._memory_chunks.values():
            doc_id = chunk.document_id
            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "document_id": doc_id,
                    "filename": chunk.metadata.get("filename", "unknown"),
                    "chunk_count": 0,
                    "metadata": chunk.metadata,
                    "created_at": datetime.now(timezone.utc),
                }
            docs_map[doc_id]["chunk_count"] += 1

        return [
            DocumentInfo(
                document_id=info["document_id"],
                filename=info["filename"],
                chunk_count=info["chunk_count"],
                metadata=info["metadata"],
                created_at=info["created_at"],
            )
            for info in docs_map.values()
        ]

    async def reset_collection(self) -> bool:
        """Drop and re-create the collection."""
        self._memory_chunks.clear()
        try:
            await self.client.delete_collection(self.collection_name)
            await init_qdrant_collection()
            return True
        except Exception as e:
            logger.warning("Error resetting collection: %s", str(e))
            return False
