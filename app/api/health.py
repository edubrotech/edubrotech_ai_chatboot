from typing import Any, Dict
from fastapi import APIRouter, status
from app.core.config import get_settings
from app.memory.redis import ChatMemory
from app.vectorstore.qdrant import check_qdrant_health

router = APIRouter(tags=["Health"])
settings = get_settings()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """Detailed health check for the service and its downstream dependencies."""
    memory = ChatMemory()
    redis_healthy = await memory.ping()
    qdrant_healthy = await check_qdrant_health()

    overall_status = "healthy"
    # If external backends are not available, report degraded rather than unhealthy (due to in-memory fallbacks)
    if not redis_healthy or not qdrant_healthy:
        overall_status = "degraded"

    return {
        "status": overall_status,
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "dependencies": {
            "redis": "connected" if redis_healthy else "fallback_in_memory",
            "qdrant": "connected" if qdrant_healthy else "fallback_in_memory",
        },
        "config": {
            "default_provider": settings.DEFAULT_LLM_PROVIDER,
            "default_model": settings.OPENAI_MODEL if settings.DEFAULT_LLM_PROVIDER == "openai" else settings.GEMINI_MODEL,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
        },
    }


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_probe() -> Dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_probe() -> Dict[str, str]:
    """Kubernetes readiness probe."""
    return {"status": "ready"}
