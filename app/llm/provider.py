from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.chat import ChatMessage

logger = get_logger(__name__)
settings = get_settings()


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, model: str, provider_name: str):
        self.model = model
        self.provider_name = provider_name

    @abstractmethod
    async def generate(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a complete text completion."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens as an asynchronous generator."""
        pass


def get_llm_provider(provider_type: Optional[str] = None) -> BaseLLMProvider:
    """
    Factory function to retrieve the configured LLM provider instance.
    Supports: "openai", "gemini", "local".
    """
    selected = (provider_type or settings.DEFAULT_LLM_PROVIDER).lower()

    if selected == "openai":
        from app.llm.openai import OpenAIProvider
        return OpenAIProvider()
    elif selected == "gemini":
        from app.llm.gemini import GeminiProvider
        return GeminiProvider()
    elif selected == "local":
        from app.llm.local import LocalLLMProvider
        return LocalLLMProvider()
    else:
        logger.warning("Unknown LLM provider '%s', falling back to OpenAI provider", selected)
        from app.llm.openai import OpenAIProvider
        return OpenAIProvider()
