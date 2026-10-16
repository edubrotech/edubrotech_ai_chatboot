from typing import AsyncGenerator, List, Optional
from openai import AsyncOpenAI
from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.provider import BaseLLMProvider
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
        if not self._has_valid_api_key():
            return (
                f"[OpenAI Provider Simulated Response] Hello! To receive live responses from {self.model}, "
                "please configure OPENAI_API_KEY in your .env file."
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
            raise

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        if not self._has_valid_api_key():
            simulated_text = (
                f"[OpenAI Provider Streaming Simulation] Please set OPENAI_API_KEY in your environment "
                f"to query {self.model}."
            )
            for token in simulated_text.split(" "):
                yield token + " "
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
            raise
