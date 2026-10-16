from typing import List, Optional
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.chat import SourceDocument
from app.rag.embeddings import BaseEmbeddingProvider, get_embedding_provider
from app.vectorstore.repository import VectorRepository

logger = get_logger(__name__)
settings = get_settings()


class RAGRetriever:
    """Retrieves relevant document chunks from the vector repository using semantic similarity."""

    def __init__(
        self,
        repository: Optional[VectorRepository] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
    ):
        self.repository = repository or VectorRepository()
        self.embedding_provider = embedding_provider or get_embedding_provider()

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SourceDocument]:
        """Generate embedding for query and retrieve top matching context chunks."""
        k = top_k if top_k is not None else settings.RAG_TOP_K
        threshold = score_threshold if score_threshold is not None else settings.RAG_SCORE_THRESHOLD

        logger.debug("Retrieving context for query: '%s' (top_k=%d, threshold=%.2f)", query, k, threshold)
        query_vector = await self.embedding_provider.embed_query(query)
        sources = await self.repository.search(
            query_vector=query_vector,
            top_k=k,
            score_threshold=threshold,
        )
        logger.info("Retrieved %d matching source chunks for query", len(sources))
        return sources
