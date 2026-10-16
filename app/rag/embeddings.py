import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import List, Optional
import google.generativeai as genai
from openai import AsyncOpenAI
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

STOP_WORDS = {
    "the", "is", "at", "which", "on", "what", "a", "an", "and", "or",
    "in", "of", "for", "to", "how", "do", "i", "my", "it", "with", "about",
    "can", "are", "you", "me", "please", "this", "that", "be", "as"
}


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


class SemanticTokenEmbeddingProvider(BaseEmbeddingProvider):
    """
    High-performance semantic token feature-hashing embedding generator.
    Produces unit-normalized dense vectors based on subword n-grams, stems, and terms.
    Ensures that semantically overlapping texts produce strong positive cosine similarity
    even completely offline with zero API keys or external model weights.
    """

    def __init__(self, dimension: int = 1536):
        super().__init__(dimension=dimension)

    def _generate_vector(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dimension

        vec = [0.0] * self.dimension
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
        if not tokens:
            tokens = [text.lower().strip()]

        terms = []
        for i, t in enumerate(tokens):
            if t not in STOP_WORDS:
                terms.append(t)
                # Sub-word stems for morphological matching (e.g., 'cancel' in 'cancellation')
                if len(t) > 4:
                    terms.append(t[:4])
                    terms.append(t[:5])
            if i < len(tokens) - 1 and tokens[i] not in STOP_WORDS and tokens[i+1] not in STOP_WORDS:
                terms.append(f"{tokens[i]}_{tokens[i+1]}")

        if not terms:
            terms = tokens

        for term in terms:
            h = int(hashlib.md5(term.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 16) & 1) else -1.0
            vec[idx] += sign

        # Unit normalize (L2 norm)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]

    async def embed_query(self, text: str) -> List[float]:
        return self._generate_vector(text)


# Backwards compatibility alias
DeterministicEmbeddingProvider = SemanticTokenEmbeddingProvider


class FastEmbedProvider(BaseEmbeddingProvider):
    """FastEmbed local neural embedding provider (ONNX CPU, 384 dimensions)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        super().__init__(dimension=384)
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
            except Exception as e:
                logger.warning("Could not initialize FastEmbed (%s), using SemanticToken fallback", str(e))
                self._model = None
        return self._model

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.model is not None:
            try:
                embeddings = list(self.model.embed(texts))
                return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]
            except Exception as e:
                logger.warning("FastEmbed embed_documents error: %s", str(e))

        fallback = SemanticTokenEmbeddingProvider(dimension=self.dimension)
        return await fallback.embed_documents(texts)

    async def embed_query(self, text: str) -> List[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI Embeddings integration (e.g. text-embedding-3-small, 1536 dim)."""

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
            logger.debug("Valid OPENAI_API_KEY not configured. Using semantic token embeddings.")
            fallback = SemanticTokenEmbeddingProvider(dimension=self.dimension)
            return await fallback.embed_documents(texts)

        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.warning("Error generating OpenAI embeddings (%s). Using semantic token fallback.", str(e))
            fallback = SemanticTokenEmbeddingProvider(dimension=self.dimension)
            return await fallback.embed_documents(texts)

    async def embed_query(self, text: str) -> List[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini Embedding API integration (e.g. models/text-embedding-004, 768 dim)."""

    def __init__(self, model: str = "models/text-embedding-004", dimension: int = 768):
        super().__init__(dimension=dimension)
        self.model = model
        self.api_key = settings.GEMINI_API_KEY
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def _is_valid_key(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("your-") and not self.api_key.startswith("sk-dummy"))

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._is_valid_key():
            logger.debug("GEMINI_API_KEY not configured. Using semantic token fallback embeddings.")
            fallback = SemanticTokenEmbeddingProvider(dimension=self.dimension)
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
            logger.warning("Error generating Gemini embeddings (%s). Using fallback.", str(e))
            fallback = SemanticTokenEmbeddingProvider(dimension=self.dimension)
            return await fallback.embed_documents(texts)

    async def embed_query(self, text: str) -> List[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]


def get_embedding_provider(provider_type: Optional[str] = None) -> BaseEmbeddingProvider:
    """Factory to retrieve configured embedding provider."""
    selected = (provider_type or settings.EMBEDDING_PROVIDER).lower()

    if selected == "openai":
        return OpenAIEmbeddingProvider()
    elif selected == "gemini":
        return GeminiEmbeddingProvider()
    elif selected == "fastembed":
        return FastEmbedProvider()
    elif selected == "local":
        return SemanticTokenEmbeddingProvider(dimension=settings.EMBEDDING_DIMENSION)
    else:
        return OpenAIEmbeddingProvider()
