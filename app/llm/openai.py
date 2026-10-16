from typing import AsyncGenerator, List, Optional
from openai import AsyncOpenAI
from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.provider import BaseLLMProvider
from app.llm.rag_responder import (
    extract_context_from_prompt,
    stream_synthesized_response,
    synthesize_rag_response,
)
from app.models.chat import ChatMessage, Role

logger = get_logger(__name__)
settings = get_settings()


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API integration for chat completions and streaming."""

    def __init__(self, model: Optional[str] = None):
        super().__init__(
            model=model or settings.OPENAI_MODEL,
            provider_name="openai",
        )
        self.api_key = settings.OPENAI_API_KEY
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            if not self.api_key:
                logger.warning("OPENAI_API_KEY is not set. Requests may fail unless mocked or configured.")
            self._client = AsyncOpenAI(api_key=self.api_key or "sk-dummy-key-for-init")
        return self._client

    def _prepare_messages(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
    ) -> List[dict]:
        payload = []
        if system_prompt:
            payload.append({"role": "system", "content": system_prompt})
        for msg in messages:
            payload.append({"role": msg.role.value, "content": msg.content})
        return payload

    def _has_valid_api_key(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("your-") and not self.api_key.startswith("sk-dummy"))

    async def generate(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        last_user_query = next((m.content for m in reversed(messages) if m.role == Role.USER), "")

        if not self._has_valid_api_key():
            context = extract_context_from_prompt(system_prompt)
            if context:
                return synthesize_rag_response(
                    query=last_user_query,
                    context=context,
                    provider_name=self.provider_name,
                    model_name=self.model,
                )
            return (
                f"[OpenAI Provider] Hello! To enable general chat reasoning with {self.model}, "
                "please configure your OPENAI_API_KEY in the .env file. For knowledge base questions, "
                "please enable RAG and ensure documents are indexed."
            )

        payload = self._prepare_messages(messages, system_prompt)
        temp = temperature if temperature is not None else settings.OPENAI_TEMPERATURE

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=temp,
                max_tokens=settings.OPENAI_MAX_TOKENS,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Error generating completion from OpenAI: %s", str(e))
            # Fallback to context-grounded response on API error
            context = extract_context_from_prompt(system_prompt)
            if context:
                logger.info("OpenAI API call failed (%s), answering from retrieved context", str(e))
                return synthesize_rag_response(
                    query=last_user_query,
                    context=context,
                    provider_name=f"{self.provider_name} (offline fallback)",
                    model_name=self.model,
                )
            raise

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        last_user_query = next((m.content for m in reversed(messages) if m.role == Role.USER), "")

        if not self._has_valid_api_key():
            context = extract_context_from_prompt(system_prompt)
            if context:
                response_text = synthesize_rag_response(
                    query=last_user_query,
                    context=context,
                    provider_name=self.provider_name,
                    model_name=self.model,
                )
            else:
                response_text = (
                    f"[OpenAI Provider] Please set OPENAI_API_KEY in your environment to query {self.model} "
                    f"for general questions, or upload documents to query the RAG knowledge base."
                )

            async for token in stream_synthesized_response(response_text):
                yield token
            return

        payload = self._prepare_messages(messages, system_prompt)
        temp = temperature if temperature is not None else settings.OPENAI_TEMPERATURE

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=temp,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                stream=True,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content if chunk.choices else None
                if content:
                    yield content
        except Exception as e:
            logger.error("Error streaming completion from OpenAI: %s", str(e))
            context = extract_context_from_prompt(system_prompt)
            if context:
                response_text = synthesize_rag_response(
                    query=last_user_query,
                    context=context,
                    provider_name=f"{self.provider_name} (offline fallback)",
                    model_name=self.model,
                )
                async for token in stream_synthesized_response(response_text):
                    yield token
            else:
                raise
