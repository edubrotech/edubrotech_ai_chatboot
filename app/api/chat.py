from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from app.core.security import verify_api_key
from app.models.chat import (
    ChatHistoryResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
)
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])


def get_chat_service() -> ChatService:
    """Dependency injector for ChatService."""
    return ChatService()


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
    summary="Send a chat message and receive a completion",
)
async def send_chat_message(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """
    Submit a prompt or question to the AI chatbot.
    Automatically maintains conversational memory and performs RAG context retrieval when enabled.
    """
    try:
        return await service.chat(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}",
        )


@router.post(
    "/stream",
    dependencies=[Depends(verify_api_key)],
    summary="Send a chat message and stream response tokens (SSE)",
)
async def stream_chat_message(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    Stream chatbot response tokens in real-time as Server-Sent Events (SSE).
    Returns 'sources', 'token', and 'done' events.
    """
    request.stream = True
    return StreamingResponse(
        service.chat_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/history/{session_id}",
    response_model=ChatHistoryResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Retrieve session chat history",
)
async def get_session_history(
    session_id: str,
    service: ChatService = Depends(get_chat_service),
) -> ChatHistoryResponse:
    """Fetch stored conversation history for a given session ID."""
    history = await service.get_history(session_id)
    return ChatHistoryResponse(
        session_id=session_id,
        messages=history,
    )


@router.delete(
    "/history/{session_id}",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
    summary="Clear session chat history",
)
async def clear_session_history(
    session_id: str,
    service: ChatService = Depends(get_chat_service),
):
    """Purge stored conversation history for a given session ID."""
    cleared = await service.clear_history(session_id)
    return {
        "session_id": session_id,
        "cleared": cleared,
        "message": "Conversation history cleared successfully",
    }
