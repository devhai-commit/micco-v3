"""
Embedding Service
=================
Generates vector embeddings using OpenAI text-embedding-3-small.

Configurable via NEXUSRAG_EMBEDDING_MODEL + OPENAI_API_KEY in settings.
"""
from __future__ import annotations

import logging
from typing import Sequence, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# Dimension lookup for OpenAI embedding models
_KNOWN_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # (cũ — local sentence-transformers, không còn dùng)
    # "BAAI/bge-m3": 1024,
    # "all-MiniLM-L6-v2": 384,
    # "intfloat/multilingual-e5-large-instruct": 1024,
}


class EmbeddingService:
    """
    Service for generating text embeddings via OpenAI API.
    Thin wrapper around OpenAIEmbeddingProvider để giữ interface cũ.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.NEXUSRAG_EMBEDDING_MODEL
        from app.services.llm.openai_provider import OpenAIEmbeddingProvider
        self._provider = OpenAIEmbeddingProvider(
            api_key=settings.OPENAI_API_KEY,
            model=self.model_name,
            base_url=settings.OPENAI_BASE_URL or None,
        )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension size."""
        return self._provider.get_dimension()

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        result = self._provider.embed_sync([text])
        return result[0].tolist()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch."""
        valid = [t for t in texts if t.strip()]
        if not valid:
            raise ValueError("All texts are empty")
        result = self._provider.embed_sync(valid)
        return result.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        return self.embed_text(query)

    async def aembed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Async batch embedding — dùng trong async context để tránh blocking."""
        valid = [t for t in texts if t.strip()]
        if not valid:
            raise ValueError("All texts are empty")
        result = await self._provider.embed(valid)
        return result.tolist()


# Default service instance (singleton)
_default_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the default embedding service."""
    global _default_service
    if _default_service is None:
        _default_service = EmbeddingService()
    return _default_service


def embed_text(text: str) -> list[float]:
    """Convenience function to embed a single text."""
    return get_embedding_service().embed_text(text)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Convenience function to embed multiple texts."""
    return get_embedding_service().embed_texts(texts)
