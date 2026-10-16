import json
import uuid
from typing import AsyncGenerator, List, Optional
from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.provider import get_llm_provider
from app.memory.redis import ChatMemory
from app.models.chat import ChatMessage, ChatRequest, ChatResponse, Role
from app.rag.chain import RAGChain
from app.rag.retriever import RAGRetriever

logger = get_logger(__name__)
settings = get_settings()


class ChatService:
    """Coordinates memory persistence, knowledge retrieval, and LLM orchestration."""

    def __init__(
        self,
        memory: Optional[ChatMemory] = None,
        retriever: Optional[RAGRetriever] = None,
    ):
        self.memory = memory or ChatMemory()
        self.retriever = retriever or RAGRetriever()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle synchronous / non-streaming conversation requests."""
        session_id = request.session_id or str(uuid.uuid4())
        user_message = ChatMessage(role=Role.USER, content=request.message)

        # Retrieve prior session messages
        history = await self.memory.get_history(session_id)
        # Append current user turn
        active_messages = history + [user_message]

        # Select provider and execute RAG chain
        llm = get_llm_provider(request.provider)
        chain = RAGChain(retriever=self.retriever, llm_provider=llm)

        assistant_content, sources = await chain.execute(
            messages=active_messages,
            system_prompt=request.system_prompt,
            use_rag=request.use_rag,
            temperature=request.temperature,
        )

        assistant_message = ChatMessage(role=Role.ASSISTANT, content=assistant_content)

        # Commit conversation turns to memory
        await self.memory.add_message(session_id, user_message)
        await self.memory.add_message(session_id, assistant_message)

        return ChatResponse(
            session_id=session_id,
            message=assistant_content,
            sources=sources,
            provider=llm.provider_name,
            model=llm.model,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """
        Handle streaming chat interactions using Server-Sent Events (SSE).
        Yields serialized JSON payloads with event types: 'sources', 'token', and 'done'.
        """
        session_id = request.session_id or str(uuid.uuid4())
        user_message = ChatMessage(role=Role.USER, content=request.message)

        history = await self.memory.get_history(session_id)
        active_messages = history + [user_message]

        llm = get_llm_provider(request.provider)
        chain = RAGChain(retriever=self.retriever, llm_provider=llm)

        token_stream, sources = await chain.execute_stream(
            messages=active_messages,
            system_prompt=request.system_prompt,
            use_rag=request.use_rag,
            temperature=request.temperature,
        )

        # Emit initial metadata and citation sources
        sources_payload = {
            "event": "sources",
            "session_id": session_id,
            "sources": [s.model_dump() for s in sources],
            "provider": llm.provider_name,
            "model": llm.model,
        }
        yield f"data: {json.dumps(sources_payload)}\n\n"

        accumulated_response: List[str] = []
        async for token in token_stream:
            accumulated_response.append(token)
            token_payload = {
                "event": "token",
                "session_id": session_id,
                "token": token,
            }
            yield f"data: {json.dumps(token_payload)}\n\n"

        full_assistant_text = "".join(accumulated_response)
        assistant_message = ChatMessage(role=Role.ASSISTANT, content=full_assistant_text)

        # Commit conversation turns to memory
        await self.memory.add_message(session_id, user_message)
        await self.memory.add_message(session_id, assistant_message)

        done_payload = {
            "event": "done",
            "session_id": session_id,
            "message": full_assistant_text,
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    async def get_history(self, session_id: str) -> List[ChatMessage]:
        """Fetch chat history for a session."""
        return await self.memory.get_history(session_id)

    async def clear_history(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        return await self.memory.clear_history(session_id)
