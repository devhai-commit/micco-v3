"""
Reranker Service
================
Cross-encoder reranking để cải thiện precision sau vector search.

Hỗ trợ hai backend:
  - Local (mặc định): BAAI/bge-reranker-v2-m3 qua sentence-transformers CrossEncoder
  - Cohere API: rerank-multilingual-v3.0 (cần COHERE_API_KEY)

Model selection:
  - Model name chứa "/" (ví dụ "BAAI/bge-reranker-v2-m3") → dùng LocalRerankerService
  - Model name không chứa "/" (ví dụ "rerank-multilingual-v3.0") → dùng CohereRerankerService
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """A single reranked item with its original index and relevance score."""
    index: int          # Original position in the input list
    score: float        # Relevance score (higher = more relevant)
    text: str           # The chunk text


# ─────────────────────────────────────────────────────────────────────────────
# Local reranker (BAAI/bge-reranker-v2-m3 via sentence-transformers)
# ─────────────────────────────────────────────────────────────────────────────

class LocalRerankerService:
    """
    Local cross-encoder reranker using sentence-transformers.
    Không cần API key, chạy hoàn toàn cục bộ.

    Recommended models:
      - BAAI/bge-reranker-v2-m3      (đa ngôn ngữ, ~568MB, tốt nhất)
      - BAAI/bge-reranker-base        (nhẹ hơn, ~278MB)
      - cross-encoder/ms-marco-MiniLM-L-6-v2  (chỉ tiếng Anh)
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.NEXUSRAG_RERANKER_MODEL
        self._model = None

    @property
    def model(self):
        """Lazy-load CrossEncoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading local reranker: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
            )
            logger.info(f"Local reranker loaded: {self.model_name}")
        return self._model

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[RerankResult]:
        """
        Rerank documents bằng cross-encoder local.

        Args:
            query: Câu hỏi của người dùng
            documents: Danh sách chunk text cần rerank
            top_k: Số kết quả tối đa trả về
            min_score: Ngưỡng score tối thiểu (None = không lọc)

        Returns:
            List RerankResult sorted by score descending.
        """
        if not documents:
            return []

        docs_list = list(documents[:100])  # giới hạn giống Cohere

        try:
            # Tạo cặp (query, doc) cho CrossEncoder
            pairs = [(query, doc) for doc in docs_list]
            scores = self.model.predict(pairs, show_progress_bar=False)

            results = [
                RerankResult(index=i, score=float(scores[i]), text=docs_list[i])
                for i in range(len(docs_list))
            ]

            # Sort by score descending
            results.sort(key=lambda r: r.score, reverse=True)

            # Lọc min_score
            if min_score is not None:
                results = [r for r in results if r.score >= min_score]

            # Giới hạn top_k
            if top_k is not None:
                results = results[:top_k]

            return results

        except Exception as e:
            logger.error(f"Local rerank failed: {e}")
            fallback = list(documents)
            if top_k is not None:
                fallback = fallback[:top_k]
            return [
                RerankResult(index=i, score=1.0 / (i + 1), text=doc)
                for i, doc in enumerate(fallback)
            ]


# ─────────────────────────────────────────────────────────────────────────────
# Cohere reranker
# ─────────────────────────────────────────────────────────────────────────────

class CohereRerankerService:
    """
    Cohere API-based reranker service.
    Cần COHERE_API_KEY. Model: rerank-multilingual-v3.0
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.NEXUSRAG_RERANKER_MODEL
        self.api_key = settings.COHERE_API_KEY
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import cohere
            if not self.api_key:
                raise ValueError("COHERE_API_KEY is not set.")
            logger.info(f"Init Cohere reranker: {self.model_name}")
            self._client = cohere.ClientV2(self.api_key)
        return self._client

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[RerankResult]:
        if not documents:
            return []

        docs_to_rerank = list(documents[:100])

        try:
            response = self.client.rerank(
                model=self.model_name,
                query=query,
                documents=docs_to_rerank,
                top_n=top_k if top_k is not None else len(docs_to_rerank),
            )
            results = [
                RerankResult(index=r.index, score=r.relevance_score, text=documents[r.index])
                for r in response.results
            ]

            if min_score is not None:
                results = [r for r in results if r.score >= min_score]

            return results

        except Exception as e:
            logger.error(f"Cohere rerank failed: {e}")
            fallback = list(documents)
            if top_k is not None:
                fallback = fallback[:top_k]
            return [
                RerankResult(index=i, score=1.0 / (i + 1), text=doc)
                for i, doc in enumerate(fallback)
            ]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

# Backward-compatible alias
RerankerService = LocalRerankerService

_default_service: Optional[LocalRerankerService | CohereRerankerService] = None


def get_reranker_service() -> LocalRerankerService | CohereRerankerService:
    """
    Factory: chọn backend dựa vào NEXUSRAG_RERANKER_MODEL.

    - Model chứa "/" (ví dụ "BAAI/bge-reranker-v2-m3") → LocalRerankerService
    - Model không chứa "/" (ví dụ "rerank-multilingual-v3.0") → CohereRerankerService
    """
    global _default_service
    if _default_service is None:
        model = settings.NEXUSRAG_RERANKER_MODEL
        if "/" in model:
            logger.info(f"Using local reranker: {model}")
            _default_service = LocalRerankerService(model)
        else:
            logger.info(f"Using Cohere reranker: {model}")
            _default_service = CohereRerankerService(model)
    return _default_service
