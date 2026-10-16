import hashlib
import math
from abc import ABC, abstractmethod
from typing import List, Optional
import google.generativeai as genai
from openai import AsyncOpenAI
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class BaseEmbeddingProvider(ABC):
    """Abstract base class for vector embedding generation."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of text chunks."""
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Compute embedding for a single search query."""
        pass


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI Embeddings integration (e.g. text-embedding-3-small)."""

    def __init__(self, model: Optional[str] = None, dimension: Optional[int] = None):
        super().__init__(dimension=dimension or settings.EMBEDDING_DIMENSION)
        self.model = model or settings.EMBEDDING_MODEL
        self.api_key = settings.OPENAI_API_KEY
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key or "sk-dummy-key-for-init")
        return self._client

    def _is_valid_key(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("your-") and not self.api_key.startswith("sk-dummy"))

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._is_valid_key():
            logger.info("Valid OPENAI_API_KEY not configured. Using deterministic fallback embeddings.")
            fallback = DeterministicEmbeddingProvider(dimension=self.dimension)
            return await fallback.embed_documents(texts)

        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.warning("Error generating OpenAI embeddings (%s). Using fallback embeddings.", str(e))
            fallback = DeterministicEmbeddingProvider(dimension=self.dimension)
            return await fallback.embed_documents(texts)

    async def embed_query(self, text: str) -> List[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini Embedding API integration (e.g. models/text-embedding-004)."""

    def __init__(self, model: str = "models/text-embedding-004", dimension: int = 768):
        super().__init__(dimension=dimension)
        self.model = model
        self.api_key = settings.GEMINI_API_KEY
        if self.api_key:
            genai.configure(api_key=self.api_key)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            logger.info("GEMINI_API_KEY not configured. Using deterministic fallback embeddings.")
            fallback = DeterministicEmbeddingProvider(dimension=self.dimension)
            return await fallback.embed_documents(texts)

        try:
            results = []
            for t in texts:
                result = genai.embed_content(
                    model=self.model,
                    content=t,
                    task_type="retrieval_document",
                )
                results.append(result["embedding"])
            return results
        except Exception as e:
            logger.error("Error generating Gemini embeddings: %s", str(e))
            raise

    async def embed_query(self, text: str) -> List[float]:
        if not self.api_key:
            fallback = DeterministicEmbeddingProvider(dimension=self.dimension)
            return await fallback.embed_query(text)

        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]


class DeterministicEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic pseudo-embedding generator.
    Produces unit-normalized dense vectors based on SHA256 character hashes.
    Guarantees consistent dimensions for offline development, local mocking, and testing.
    """

    def __init__(self, dimension: int = 1536):
        super().__init__(dimension=dimension)

    def _generate_vector(self, text: str) -> List[float]:
        # Hash text to generate repeatable vector components
        vec = []
        seed = text.encode("utf-8")
        for i in range(self.dimension):
            h = hashlib.sha256(seed + i.to_bytes(4, byteorder="big")).digest()
            val = (int.from_bytes(h[:4], "big") / 0xFFFFFFFF) * 2.0 - 1.0
            vec.append(val)

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]

    async def embed_query(self, text: str) -> List[float]:
        return self._generate_vector(text)


def get_embedding_provider(provider_type: Optional[str] = None) -> BaseEmbeddingProvider:
    """Factory to retrieve configured embedding provider."""
    selected = (provider_type or settings.EMBEDDING_PROVIDER).lower()

    if selected == "openai":
        return OpenAIEmbeddingProvider()
    elif selected == "gemini":
        return GeminiEmbeddingProvider()
    elif selected == "local":
        return DeterministicEmbeddingProvider(dimension=settings.EMBEDDING_DIMENSION)
    else:
        return OpenAIEmbeddingProvider()
