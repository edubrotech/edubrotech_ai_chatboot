from contextlib import asynccontextmanager
from typing import Any, Dict
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api import chat, documents, health
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.vectorstore.qdrant import init_qdrant_collection

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    setup_logging()
    logger.info("Initializing %s (v%s) in %s mode...", settings.PROJECT_NAME, settings.VERSION, settings.ENVIRONMENT)

    # Initialize vector collection in background/startup
    try:
        await init_qdrant_collection()
    except Exception as e:
        logger.warning("Could not auto-initialize Qdrant on startup: %s", str(e))

    yield

    logger.info("Shutting down %s...", settings.PROJECT_NAME)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-grade AI Chatbot Service with RAG, Multi-Provider LLMs, and Redis Memory.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming HTTP request paths and method."""
    logger.debug("Request started: %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.debug("Request finished: %s %s [status: %d]", request.method, request.url.path, response.status_code)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all unhandled exception handler to return clean JSON error payloads."""
    logger.error("Unhandled server exception on %s %s: %s", request.method, request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please check logs for details."},
    )


# Include health endpoints at root level
app.include_router(health.router)

# Include versioned API endpoints
app.include_router(chat.router, prefix=settings.API_V1_STR)
app.include_router(documents.router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """Root metadata endpoint."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "api_v1": settings.API_V1_STR,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
