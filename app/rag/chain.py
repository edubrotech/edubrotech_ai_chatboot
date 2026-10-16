from typing import AsyncGenerator, List, Optional, Tuple
from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.provider import BaseLLMProvider, get_llm_provider
from app.models.chat import ChatMessage, SourceDocument
from app.rag.retriever import RAGRetriever

logger = get_logger(__name__)
settings = get_settings()

DEFAULT_SYSTEM_PROMPT = (
    "You are an intelligent, accurate, and professional AI assistant. "
    "When context is provided below, prioritize it to answer the user query faithfully. "
    "If the context does not provide the answer, provide the most helpful response possible "
    "and acknowledge the absence of direct context."
)


class RAGChain:
    """Orchestrates document context retrieval, prompt augmentation, and LLM response generation."""

    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
    ):
        self.retriever = retriever or RAGRetriever()
        self.llm_provider = llm_provider or get_llm_provider()

    def format_context(self, sources: List[SourceDocument]) -> str:
        """Format retrieved source chunks into a formatted text block for LLM conditioning."""
        if not sources:
            return ""

        context_parts = ["--- RELEVANT KNOWLEDGE BASE CONTEXT ---"]
        for idx, src in enumerate(sources, 1):
            filename = src.metadata.get("filename", f"Doc {src.document_id}")
            context_parts.append(
                f"[Source {idx} | File: {filename} | Score: {src.score:.2f}]\n{src.content}"
            )
        context_parts.append("--- END OF CONTEXT ---")
        return "\n\n".join(context_parts)

    def build_system_prompt(
        self,
        base_prompt: Optional[str],
        sources: List[SourceDocument],
    ) -> str:
        """Combine base system instructions with formatted retrieval context."""
        prompt = base_prompt or DEFAULT_SYSTEM_PROMPT
        if sources:
            context_str = self.format_context(sources)
            prompt = f"{prompt}\n\n{context_str}\n\nPlease ground your response in the provided context above."
        return prompt

    async def execute(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        use_rag: bool = True,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, List[SourceDocument]]:
        """Run complete RAG workflow: retrieve -> augment -> generate -> return text and sources."""
        sources: List[SourceDocument] = []

        if use_rag and messages:
            last_user_query = next((m.content for m in reversed(messages) if m.role == "user"), "")
            if last_user_query:
                try:
                    sources = await self.retriever.retrieve(query=last_user_query, top_k=top_k)
                except Exception as e:
                    logger.error("RAG retrieval failed, proceeding without context: %s", str(e))

        effective_prompt = self.build_system_prompt(system_prompt, sources)
        response_text = await self.llm_provider.generate(
            messages=messages,
            system_prompt=effective_prompt,
            temperature=temperature,
        )
        return response_text, sources

    async def execute_stream(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        use_rag: bool = True,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[AsyncGenerator[str, None], List[SourceDocument]]:
        """Run streaming RAG workflow: retrieve sources upfront, then stream generated tokens."""
        sources: List[SourceDocument] = []

        if use_rag and messages:
            last_user_query = next((m.content for m in reversed(messages) if m.role == "user"), "")
            if last_user_query:
                try:
                    sources = await self.retriever.retrieve(query=last_user_query, top_k=top_k)
                except Exception as e:
                    logger.error("RAG retrieval failed for stream: %s", str(e))

        effective_prompt = self.build_system_prompt(system_prompt, sources)
        token_stream = self.llm_provider.generate_stream(
            messages=messages,
            system_prompt=effective_prompt,
            temperature=temperature,
        )
        return token_stream, sources
