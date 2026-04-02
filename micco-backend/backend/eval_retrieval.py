"""
eval_retrieval.py
=================
Tự động đánh giá chất lượng truy xuất của NexusRAG.

Pipeline:
  1. Lấy tất cả chunks từ ChromaDB (workspace đã chỉ định)
  2. Sample N chunks ngẫu nhiên làm "ground truth source"
  3. Dùng LLM sinh câu hỏi cho mỗi chunk → (question, relevant_chunk_ids)
  4. Chạy retrieval theo 3 chế độ: vector-only, vector+rerank, hybrid(KG+vector+rerank)
  5. Tính Precision@K, Recall@K, MRR@K và in báo cáo

Chạy:
    cd micco-backend/backend
    python eval_retrieval.py --workspace-id 1 --sample 50 --top-k 5

Yêu cầu:
    - ChromaDB đang chạy và đã có dữ liệu (workspace đã upload tài liệu)
    - .env đã cấu hình LLM_PROVIDER, GOOGLE_AI_API_KEY / OPENAI_API_KEY
    - COHERE_API_KEY nếu muốn đánh giá chế độ rerank
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Thêm backend vào sys.path ────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("eval")

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_SAMPLE_SIZE = 30
DEFAULT_TOP_K = 5
DEFAULT_WORKSPACE_ID = 1
QUESTIONS_PER_CHUNK = 2          # số câu hỏi LLM sinh cho mỗi chunk
MIN_CHUNK_LENGTH = 80            # bỏ qua chunk quá ngắn


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    """Một cặp (câu hỏi, ground truth chunk)."""
    question: str
    relevant_chunk_id: str          # ID chunk gốc sinh ra câu hỏi
    relevant_chunk_text: str        # Nội dung chunk gốc (để debug)
    source_doc: str = ""


@dataclass
class RetrievalResult:
    """Kết quả retrieval cho một câu hỏi."""
    question: str
    retrieved_ids: list[str]        # Danh sách chunk ID trả về (theo thứ tự)
    latency_ms: float = 0.0


@dataclass
class MetricScore:
    """Điểm cho một test case."""
    precision: float
    recall: float
    reciprocal_rank: float


@dataclass
class EvalReport:
    """Báo cáo tổng hợp cho một chế độ retrieval."""
    mode: str
    k: int
    num_cases: int
    mean_precision: float
    mean_recall: float
    mrr: float
    avg_latency_ms: float
    per_case: list[dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Bước 1: Lấy chunks từ ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_chunks(workspace_id: int) -> list[dict]:
    """
    Trả về list of {id, text, metadata} từ ChromaDB collection của workspace.
    """
    from app.services.vector_store import get_vector_store

    store = get_vector_store(workspace_id)
    collection = store.collection

    total = collection.count()
    if total == 0:
        raise RuntimeError(
            f"Workspace {workspace_id} không có chunk nào trong ChromaDB. "
            "Hãy upload tài liệu trước."
        )

    # Lấy tất cả (ChromaDB hỗ trợ get() không cần embedding)
    result = collection.get(include=["documents", "metadatas"])

    chunks = []
    for idx, chunk_id in enumerate(result["ids"]):
        text = result["documents"][idx] if result.get("documents") else ""
        meta = result["metadatas"][idx] if result.get("metadatas") else {}
        if len(text.strip()) >= MIN_CHUNK_LENGTH:
            chunks.append({"id": chunk_id, "text": text, "metadata": meta})

    logger.warning(f"Lấy được {len(chunks)}/{total} chunks (sau lọc độ dài) từ workspace {workspace_id}")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Bước 2: Sinh câu hỏi từ chunk bằng LLM
# ─────────────────────────────────────────────────────────────────────────────

_QUESTION_GEN_PROMPT = """\
Dưới đây là một đoạn văn bản từ tài liệu. Hãy sinh ra {n} câu hỏi ngắn gọn bằng tiếng Việt \
mà đoạn này có thể trả lời được.

Quy tắc:
- Mỗi câu hỏi trên một dòng, bắt đầu bằng dấu "-"
- Câu hỏi phải cụ thể, không hỏi chung chung kiểu "Đoạn này nói về gì?"
- Không lặp lại nguyên văn nội dung đoạn
- Nếu đoạn không đủ thông tin để đặt câu hỏi, trả về "SKIP"

Đoạn văn:
\"\"\"
{chunk_text}
\"\"\"

Câu hỏi:"""


async def generate_questions_for_chunk(
    chunk_text: str,
    llm,
    n: int = QUESTIONS_PER_CHUNK,
) -> list[str]:
    """Dùng LLM sinh n câu hỏi từ chunk_text. Trả về danh sách câu hỏi."""
    from app.services.llm.types import LLMMessage

    prompt = _QUESTION_GEN_PROMPT.format(
        n=n,
        chunk_text=chunk_text[:1500],   # cắt để tiết kiệm token
    )
    try:
        response = await llm.acomplete(
            [LLMMessage(role="user", content=prompt)],
            temperature=0.3,
            max_tokens=512,
        )
        text = response if isinstance(response, str) else response.content

        if "SKIP" in text.upper():
            return []

        questions = []
        for line in text.splitlines():
            line = line.strip().lstrip("-• ").strip()
            if line and len(line) > 10 and "?" in line:
                questions.append(line)

        return questions[:n]
    except Exception as e:
        logger.warning(f"LLM sinh câu hỏi thất bại: {e}")
        return []


async def build_test_cases(
    chunks: list[dict],
    sample_size: int,
    questions_per_chunk: int = QUESTIONS_PER_CHUNK,
) -> list[TestCase]:
    """
    Sample chunks ngẫu nhiên, dùng LLM sinh câu hỏi → danh sách TestCase.
    """
    from app.services.llm import get_llm_provider

    llm = get_llm_provider()

    # Sample ngẫu nhiên
    sampled = random.sample(chunks, min(sample_size, len(chunks)))

    test_cases: list[TestCase] = []
    total = len(sampled)

    print(f"\n[1/3] Đang sinh câu hỏi cho {total} chunks...")

    for i, chunk in enumerate(sampled, 1):
        print(f"  → Chunk {i}/{total} (id={chunk['id'][:30]}...)", end="\r")

        questions = await generate_questions_for_chunk(
            chunk["text"], llm, n=questions_per_chunk
        )

        for q in questions:
            test_cases.append(TestCase(
                question=q,
                relevant_chunk_id=chunk["id"],
                relevant_chunk_text=chunk["text"],
                source_doc=chunk["metadata"].get("source", ""),
            ))

        # Thêm delay nhỏ để tránh rate limit
        await asyncio.sleep(0.3)

    print(f"\n  ✓ Sinh được {len(test_cases)} câu hỏi từ {total} chunks")
    return test_cases


# ─────────────────────────────────────────────────────────────────────────────
# Bước 3: Chạy retrieval
# ─────────────────────────────────────────────────────────────────────────────

def run_vector_retrieval(
    question: str,
    workspace_id: int,
    top_k: int,
) -> tuple[list[str], float]:
    """
    Vector-only retrieval (không rerank, không KG).
    Trả về (list chunk_ids, latency_ms).
    """
    from app.services.embedder import get_embedding_service
    from app.services.vector_store import get_vector_store

    embedder = get_embedding_service()
    store = get_vector_store(workspace_id)

    t0 = time.perf_counter()
    query_emb = embedder.embed_query(question)
    results = store.query(query_embedding=query_emb, n_results=top_k)
    latency = (time.perf_counter() - t0) * 1000

    ids = results.get("ids", [])
    return ids, latency




def run_vector_rerank_retrieval(
    question: str,
    workspace_id: int,
    top_k: int,
    prefetch: int = 20,
) -> tuple[list[str], float]:
    """
    Vector over-fetch → Cohere rerank.
    Trả về (list chunk_ids, latency_ms).
    """
    from app.services.embedder import get_embedding_service
    from app.services.vector_store import get_vector_store
    from app.services.reranker import get_reranker_service

    embedder = get_embedding_service()
    store = get_vector_store(workspace_id)
    reranker = get_reranker_service()

    t0 = time.perf_counter()

    # Over-fetch
    query_emb = embedder.embed_query(question)
    results = store.query(query_embedding=query_emb, n_results=prefetch)

    raw_ids = results.get("ids", [])
    raw_docs = results.get("documents", [])

    if not raw_docs:
        return raw_ids[:top_k], (time.perf_counter() - t0) * 1000

    # Rerank — dùng cùng min_score với DeepRetriever để so sánh công bằng
    from app.core.config import settings as _s
    reranked = reranker.rerank(
        question, raw_docs, top_k=top_k,
        min_score=_s.NEXUSRAG_MIN_RELEVANCE_SCORE,
    )
    reranked_ids = [raw_ids[r.index] for r in reranked if r.index < len(raw_ids)]

    latency = (time.perf_counter() - t0) * 1000
    return reranked_ids, latency


async def run_hybrid_retrieval(
    question: str,
    workspace_id: int,
    top_k: int,
) -> tuple[list[str], float]:
    """
    NexusRAG hybrid: KG + Vector + Rerank (query_deep).
    Trả về (list chunk_ids, latency_ms).
    Nếu COHERE_API_KEY không có, reranker bị bỏ qua (fallback score-based).
    """
    from app.services.nexus_rag_service import NexusRAGService
    from app.core.config import settings

    # Khởi tạo NexusRAGService — db=None bỏ qua image/table lookup
    svc = NexusRAGService(db=None, workspace_id=workspace_id)

    t0 = time.perf_counter()
    try:
        deep_result = await svc.retriever.query(
            question=question,
            mode=settings.NEXUSRAG_DEFAULT_QUERY_MODE,
            top_k=top_k,
            include_images=False,
        )
        # EnrichedChunk không có chunk_id — tái tạo từ document_id + chunk_index
        # theo format đã dùng khi index: "doc_{document_id}_chunk_{chunk_index}"
        chunk_ids = [
            f"doc_{c.document_id}_chunk_{c.chunk_index}"
            for c in deep_result.chunks
        ]
    except Exception as e:
        logger.warning(f"Hybrid retrieval lỗi: {e}, fallback vector-only")
        chunk_ids, _ = run_vector_retrieval(question, workspace_id, top_k)

    latency = (time.perf_counter() - t0) * 1000
    return chunk_ids, latency


# ─────────────────────────────────────────────────────────────────────────────
# Bước 4: Tính metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    retrieved_ids: list[str],
    relevant_id: str,
    k: int,
) -> MetricScore:
    """
    Tính Precision@K, Recall@K, Reciprocal Rank cho một test case.

    Ground truth = 1 chunk (chunk gốc sinh ra câu hỏi).
    Relevant set = {relevant_id}.
    """
    top_k = retrieved_ids[:k]
    hit = relevant_id in top_k

    precision = (1 / k) if hit else 0.0
    recall = 1.0 if hit else 0.0

    # Reciprocal rank: 1/rank_of_first_hit
    rr = 0.0
    for rank, rid in enumerate(top_k, start=1):
        if rid == relevant_id:
            rr = 1.0 / rank
            break

    return MetricScore(precision=precision, recall=recall, reciprocal_rank=rr)


def aggregate_metrics(
    mode: str,
    k: int,
    results: list[tuple[TestCase, RetrievalResult]],
) -> EvalReport:
    scores = [
        compute_metrics(r.retrieved_ids, tc.relevant_chunk_id, k)
        for tc, r in results
    ]

    n = len(scores)
    mean_p = sum(s.precision for s in scores) / n
    mean_r = sum(s.recall for s in scores) / n
    mrr = sum(s.reciprocal_rank for s in scores) / n
    avg_lat = sum(r.latency_ms for _, r in results) / n

    per_case = []
    for (tc, r), s in zip(results, scores):
        per_case.append({
            "question": tc.question,
            "source_doc": tc.source_doc,
            "relevant_id": tc.relevant_chunk_id,
            "retrieved_top1": r.retrieved_ids[0] if r.retrieved_ids else "",
            "hit": s.recall > 0,
            "rank": next(
                (i + 1 for i, rid in enumerate(r.retrieved_ids[:k])
                 if rid == tc.relevant_chunk_id),
                None,
            ),
            "latency_ms": round(r.latency_ms, 1),
        })

    return EvalReport(
        mode=mode,
        k=k,
        num_cases=n,
        mean_precision=round(mean_p, 4),
        mean_recall=round(mean_r, 4),
        mrr=round(mrr, 4),
        avg_latency_ms=round(avg_lat, 1),
        per_case=per_case,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bước 5: In báo cáo
# ─────────────────────────────────────────────────────────────────────────────

def print_report(reports: list[EvalReport], k: int) -> None:
    """In bảng so sánh các chế độ retrieval."""
    sep = "─" * 72

    print(f"\n{'=' * 72}")
    print(f"  RETRIEVAL EVALUATION REPORT  |  K = {k}")
    print(f"{'=' * 72}")
    print(f"  {'Mode':<28} {'P@K':>8} {'R@K':>8} {'MRR@K':>8} {'Lat(ms)':>10}")
    print(sep)

    for rep in reports:
        print(
            f"  {rep.mode:<28} "
            f"{rep.mean_precision:>8.4f} "
            f"{rep.mean_recall:>8.4f} "
            f"{rep.mrr:>8.4f} "
            f"{rep.avg_latency_ms:>10.1f}"
        )

    print(sep)
    print(f"  Số câu hỏi đánh giá: {reports[0].num_cases if reports else 0}")
    print(f"{'=' * 72}\n")

    # Chi tiết miss cases cho chế độ tốt nhất
    best = max(reports, key=lambda r: r.mrr)
    missed = [c for c in best.per_case if not c["hit"]]
    if missed:
        print(f"[{best.mode}] Câu hỏi không tìm thấy ({len(missed)} cases):")
        for c in missed[:5]:
            print(f"  • {c['question']}")
            print(f"    Source: {c['source_doc']}")
        if len(missed) > 5:
            print(f"  ... và {len(missed) - 5} câu khác")
        print()


def save_report_json(reports: list[EvalReport], output_path: Path) -> None:
    data = []
    for rep in reports:
        data.append({
            "mode": rep.mode,
            "k": rep.k,
            "num_cases": rep.num_cases,
            "mean_precision": rep.mean_precision,
            "mean_recall": rep.mean_recall,
            "mrr": rep.mrr,
            "avg_latency_ms": rep.avg_latency_ms,
            "per_case": rep.per_case,
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu báo cáo chi tiết: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def run_evaluation(
    workspace_id: int,
    sample_size: int,
    top_k: int,
    modes: list[str],
    seed: Optional[int],
    output: Optional[Path],
) -> None:
    if seed is not None:
        random.seed(seed)

    # ── Bước 1: Lấy chunks ──────────────────────────────────────────────────
    print(f"\n[0/3] Kết nối ChromaDB workspace {workspace_id}...")
    chunks = fetch_all_chunks(workspace_id)

    # ── Bước 2: Sinh câu hỏi ────────────────────────────────────────────────
    test_cases = await build_test_cases(chunks, sample_size)

    if not test_cases:
        print("Không sinh được câu hỏi nào. Kiểm tra lại LLM config.")
        return

    # ── Bước 3: Chạy retrieval ───────────────────────────────────────────────
    print(f"\n[2/3] Chạy retrieval ({len(test_cases)} câu hỏi, top-{top_k})...")

    reports: list[EvalReport] = []

    # Chế độ vector-only
    if "vector" in modes:
        print("  → Vector-only...")
        results_v = []
        for tc in test_cases:
            ids, lat = run_vector_retrieval(tc.question, workspace_id, top_k)
            results_v.append((tc, RetrievalResult(tc.question, ids, lat)))
        reports.append(aggregate_metrics("vector_only", top_k, results_v))

    # Chế độ vector + rerank
    if "rerank" in modes:
        from app.core.config import settings
        reranker_label = f"vector_rerank ({settings.NEXUSRAG_RERANKER_MODEL.split('/')[-1]})"
        print(f"  → Vector + Rerank [{settings.NEXUSRAG_RERANKER_MODEL}]...")
        results_r = []
        for tc in test_cases:
            try:
                ids, lat = run_vector_rerank_retrieval(tc.question, workspace_id, top_k)
            except Exception as e:
                logger.warning(f"Rerank lỗi ({e}), fallback vector")
                ids, lat = run_vector_retrieval(tc.question, workspace_id, top_k)
            results_r.append((tc, RetrievalResult(tc.question, ids, lat)))
        reports.append(aggregate_metrics(reranker_label, top_k, results_r))

    # Chế độ hybrid (KG + Vector + Rerank)
    if "hybrid" in modes:
        print("  → Hybrid (KG + Vector + Rerank)...")
        results_h = []
        for i, tc in enumerate(test_cases):
            print(f"     {i+1}/{len(test_cases)}", end="\r")
            ids, lat = await run_hybrid_retrieval(tc.question, workspace_id, top_k)
            results_h.append((tc, RetrievalResult(tc.question, ids, lat)))
            await asyncio.sleep(0.1)
        print()
        reports.append(aggregate_metrics("hybrid_nexusrag", top_k, results_h))

    # ── Bước 4: In và lưu báo cáo ────────────────────────────────────────────
    print("\n[3/3] Tổng hợp kết quả...")
    print_report(reports, top_k)

    if output:
        save_report_json(reports, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đánh giá chất lượng retrieval của NexusRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workspace-id", type=int, default=DEFAULT_WORKSPACE_ID,
        help=f"ID workspace/knowledge base cần đánh giá (mặc định: {DEFAULT_WORKSPACE_ID})",
    )
    parser.add_argument(
        "--sample", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Số chunk lấy mẫu để sinh câu hỏi (mặc định: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Số kết quả truy xuất (mặc định: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--modes", nargs="+",
        choices=["vector", "rerank", "hybrid"],
        default=["vector", "rerank", "hybrid"],
        help="Các chế độ cần đánh giá (mặc định: tất cả)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed để tái lập kết quả (mặc định: 42)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Lưu báo cáo chi tiết ra file JSON (ví dụ: eval_report.json)",
    )
    parser.add_argument(
        "--no-hybrid", action="store_true",
        help="Bỏ qua chế độ hybrid (dùng khi KG chưa được xây dựng)",
    )

    args = parser.parse_args()

    modes = args.modes
    if args.no_hybrid and "hybrid" in modes:
        modes = [m for m in modes if m != "hybrid"]

    print("=" * 72)
    print("  NexusRAG Retrieval Evaluator")
    print("=" * 72)
    print(f"  Workspace ID : {args.workspace_id}")
    print(f"  Sample size  : {args.sample} chunks × {QUESTIONS_PER_CHUNK} câu hỏi")
    print(f"  Top-K        : {args.top_k}")
    print(f"  Modes        : {', '.join(modes)}")
    print(f"  Random seed  : {args.seed}")

    asyncio.run(
        run_evaluation(
            workspace_id=args.workspace_id,
            sample_size=args.sample,
            top_k=args.top_k,
            modes=modes,
            seed=args.seed,
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()
