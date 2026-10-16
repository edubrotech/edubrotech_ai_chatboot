from typing import Optional
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as qmodels
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

_async_client: Optional[AsyncQdrantClient] = None
_sync_client: Optional[QdrantClient] = None


def get_sync_qdrant_client() -> QdrantClient:
    """Return synchronous Qdrant client instance."""
    global _sync_client
    if _sync_client is None:
        _sync_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY or None,
            prefer_grpc=settings.QDRANT_PREFER_GRPC,
            timeout=5.0,
        )
    return _sync_client


def get_async_qdrant_client() -> AsyncQdrantClient:
    """Return asynchronous Qdrant client instance."""
    global _async_client
    if _async_client is None:
        _async_client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY or None,
            prefer_grpc=settings.QDRANT_PREFER_GRPC,
            timeout=5.0,
        )
    return _async_client


async def init_qdrant_collection() -> bool:
    """
    Ensure the target collection exists with proper vector dimension and cosine metric.
    Returns True if collection exists or was created, False if Qdrant is unreachable.
    """
    client = get_async_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION

    try:
        collections_response = await client.get_collections()
        existing = [c.name for c in collections_response.collections]

        if collection_name not in existing:
            logger.info("Creating Qdrant collection: %s", collection_name)
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info("Successfully created Qdrant collection: %s", collection_name)
        else:
            logger.debug("Qdrant collection '%s' already exists.", collection_name)
        return True
    except Exception as e:
        logger.warning("Could not initialize Qdrant collection (%s): %s", collection_name, str(e))
        return False


async def check_qdrant_health() -> bool:
    """Check if Qdrant server is healthy and responding."""
    client = get_async_qdrant_client()
    try:
        await client.get_collections()
        return True
    except Exception:
        return False
