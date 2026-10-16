from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from app.core.logging import get_logger
from app.core.security import verify_api_key
from app.models.document import (
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentUploadResponse,
)
from app.rag.loader import DocumentLoader
from app.services.document_service import DocumentService

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


def get_document_service() -> DocumentService:
    """Dependency injector for DocumentService."""
    return DocumentService()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_api_key)],
    summary="Upload and index a document",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file (.txt, .md, .pdf, .json, .csv)"),
    service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    """
    Upload a document into the knowledge base.
    The file will be parsed, chunked, embedded, and indexed into the vector store.
    """
    filename = file.filename or "uploaded_file"
    ext = Path(filename).suffix.lower()

    if ext not in DocumentLoader.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Supported formats: {', '.join(DocumentLoader.SUPPORTED_EXTENSIONS)}",
        )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        response = await service.ingest_bytes(filename=filename, content=content)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error("Failed to ingest file %s: %s", filename, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}",
        )


@router.get(
    "",
    response_model=List[DocumentInfo],
    dependencies=[Depends(verify_api_key)],
    summary="List all indexed documents",
)
async def list_documents(
    service: DocumentService = Depends(get_document_service),
) -> List[DocumentInfo]:
    """Retrieve metadata and chunk counts for all currently indexed documents."""
    return await service.list_documents()


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Delete a document and its embeddings",
)
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentDeleteResponse:
    """Purge a document and its vector embeddings from the vector store."""
    return await service.delete_document(document_id)
