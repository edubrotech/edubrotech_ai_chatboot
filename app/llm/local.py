from typing import AsyncGenerator, List, Optional
import httpx
from openai import AsyncOpenAI
from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.provider import BaseLLMProvider
from app.models.chat import ChatMessage

logger = get_logger(__name__)
settings = get_settings()


class LocalLLMProvider(BaseLLMProvider):
    """Local LLM provider (Ollama, vLLM, LM Studio) using OpenAI-compatible HTTP interface."""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(
            model=model or settings.LOCAL_LLM_MODEL,
            provider_name="local",
        )
        self.base_url = base_url or settings.LOCAL_LLM_URL
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key="local-token",
                timeout=60.0,
            )
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

    async def generate(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        payload = self._prepare_messages(messages, system_prompt)
        temp = temperature if temperature is not None else 0.7

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=temp,
                max_tokens=settings.OPENAI_MAX_TOKENS,
            )
            return response.choices[0].message.content or ""
        except (httpx.ConnectError, Exception) as e:
            logger.warning("Local LLM at %s unreachable (%s), providing fallback response", self.base_url, str(e))
            return (
                f"[Local LLM Provider Response] Connected to fallback mode. Ensure Ollama or local LLM is running "
                f"at {self.base_url} with model '{self.model}'."
            )

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        payload = self._prepare_messages(messages, system_prompt)
        temp = temperature if temperature is not None else 0.7

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
        except (httpx.ConnectError, Exception) as e:
            logger.warning("Local LLM stream failed: %s", str(e))
            fallback_text = (
                f"[Local LLM Fallback Stream] Could not connect to local LLM server at {self.base_url}."
            )
            for token in fallback_text.split(" "):
                yield token + " "
