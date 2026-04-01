"""
OpenAI LLM & Embedding Providers
==================================
Concrete implementations using the ``openai`` Python library.
Supports: gpt-4o, gpt-4o-mini, gpt-4.1, o1, o3-mini, text-embedding-3-small/large.
"""
from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

import numpy as np

from app.services.llm.base import EmbeddingProvider, LLMProvider
from app.services.llm.types import LLMMessage, LLMResult, StreamChunk

logger = logging.getLogger(__name__)

# Models that support vision (image input)
_VISION_MODELS = {
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview",
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
}

# Embedding model → dimension mapping
_EMBEDDING_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _to_openai_messages(
    messages: list[LLMMessage],
    system_prompt: Optional[str] = None,
) -> list[dict]:
    """Convert LLMMessage list → OpenAI message dicts."""
    result: list[dict] = []

    if system_prompt:
        result.append({"role": "system", "content": system_prompt})

    for msg in messages:
        if msg.images:
            import base64
            content = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for img in msg.images:
                b64 = base64.b64encode(img.data).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img.mime_type};base64,{b64}"},
                })
            result.append({"role": msg.role, "content": content})
        else:
            result.append({"role": msg.role, "content": msg.content})

    return result


class OpenAILLMProvider(LLMProvider):
    """OpenAI text/multimodal generation."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        max_output_tokens: int = 8192,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._max_output_tokens = max_output_tokens

    def _get_client(self):
        from openai import OpenAI
        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    def _get_async_client(self):
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        client = self._get_client()
        oai_msgs = _to_openai_messages(messages, system_prompt)

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=oai_msgs,
                temperature=temperature,
                max_tokens=min(max_tokens, self._max_output_tokens),
            )
            content = response.choices[0].message.content or ""
            return content
        except Exception as e:
            logger.error(f"OpenAI LLM call failed: {e}", exc_info=True)
            return ""

    async def acomplete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        client = self._get_async_client()
        oai_msgs = _to_openai_messages(messages, system_prompt)

        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=oai_msgs,
                temperature=temperature,
                max_tokens=min(max_tokens, self._max_output_tokens),
            )
            content = response.choices[0].message.content or ""
            return content
        except Exception as e:
            logger.error(f"OpenAI async LLM call failed: {e}", exc_info=True)
            return ""

    async def astream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
        tools: list | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        client = self._get_async_client()
        oai_msgs = _to_openai_messages(messages, system_prompt)

        kwargs: dict = dict(
            model=self._model,
            messages=oai_msgs,
            temperature=temperature,
            max_tokens=min(max_tokens, self._max_output_tokens),
            stream=True,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                # Tool call chunks
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.function:
                            yield StreamChunk(
                                type="function_call",
                                function_call={
                                    "name": tc.function.name or "",
                                    "args": tc.function.arguments or "",
                                },
                            )
                    continue

                if delta.content:
                    yield StreamChunk(type="text", text=delta.content)

        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}", exc_info=True)
            yield StreamChunk(type="text", text="")

    def supports_vision(self) -> bool:
        return any(v in self._model for v in ("gpt-4o", "gpt-4-turbo", "gpt-4.1", "gpt-4-vision"))

    def supports_thinking(self) -> bool:
        return False


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text embedding (text-embedding-3-small / text-embedding-3-large)."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._dimension: Optional[int] = None

    def _get_client(self):
        from openai import OpenAI
        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    def embed_sync(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        # OpenAI embedding: truncate empty strings to avoid API errors
        clean = [t.strip() or "[empty]" for t in texts]

        try:
            response = client.embeddings.create(model=self._model, input=clean)
            vectors = [item.embedding for item in response.data]
            return np.array(vectors, dtype=np.float32)
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            dim = self.get_dimension()
            return np.zeros((len(texts), dim), dtype=np.float32)

    async def embed(self, texts: list[str]) -> np.ndarray:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        clean = [t.strip() or "[empty]" for t in texts]

        try:
            response = await client.embeddings.create(model=self._model, input=clean)
            vectors = [item.embedding for item in response.data]
            return np.array(vectors, dtype=np.float32)
        except Exception as e:
            logger.error(f"OpenAI async embedding failed: {e}")
            dim = self.get_dimension()
            return np.zeros((len(texts), dim), dtype=np.float32)

    def get_dimension(self) -> int:
        if self._dimension is None:
            self._dimension = _EMBEDDING_DIMS.get(self._model, 1536)
        return self._dimension
