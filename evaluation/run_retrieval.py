import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import settings
from app.services.document_pipeline import DocumentPipeline
from app.services.document_service import DocumentService


def average_precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    hits = 0
    precision_sum = 0.0
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant), k) if relevant else 0.0


def evaluate_rankings(dataset: dict, rankings: dict[str, list[str]], ks: tuple[int, ...]) -> dict:
    question_metrics = {}
    for item in dataset["questions"]:
        query = item["question"]
        relevant = set(item["relevant_chunk_ids"])
        ranked = rankings[query]
        reciprocal_rank = next(
            (1 / rank for rank, chunk_id in enumerate(ranked, start=1) if chunk_id in relevant),
            0.0,
        )
        values = {"mrr": reciprocal_rank}
        for k in ks:
            retrieved = ranked[:k]
            values[f"recall@{k}"] = len(set(retrieved) & relevant) / len(relevant)
            values[f"precision@{k}"] = len(set(retrieved) & relevant) / k
            values[f"ndcg@{k}"] = (
                sum(1 / np.log2(rank + 1) for rank, chunk_id in enumerate(retrieved, start=1) if chunk_id in relevant)
                / sum(1 / np.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
            )
        question_metrics[query] = values
    metric_names = next(iter(question_metrics.values())).keys()
    aggregate = {
        name: float(np.mean([values[name] for values in question_metrics.values()]))
        for name in metric_names
    }
    return {"aggregate": aggregate, "per_question": question_metrics}


def evaluate_grounding(dataset: dict, rankings: dict[str, list[str]], dense_scores: dict[str, dict[str, float]]) -> dict:
    chunk_ids = {chunk["id"] for chunk in dataset["chunks"]}
    answerable = dataset["questions"]
    supported = []
    citation_valid = []
    for item in answerable:
        question = item["question"]
        relevant = set(item["relevant_chunk_ids"])
        cited = rankings[question][:3]
        supported.append(bool(set(cited) & relevant))
        citation_valid.extend(chunk_id in chunk_ids for chunk_id in cited)
    unsupported_decisions = []
    for item in dataset.get("unsupported_questions", []):
        scores = dense_scores[item["question"]]
        has_evidence = any(score >= settings.DENSE_RELEVANCE_THRESHOLD for score in scores.values())
        unsupported_decisions.append(not has_evidence)
    return {
        "supported_answerable_query_rate_at_3": float(np.mean(supported)) if supported else 0.0,
        "citation_record_integrity_rate": float(np.mean(citation_valid)) if citation_valid else 0.0,
        "unsupported_query_abstention_rate": float(np.mean(unsupported_decisions)) if unsupported_decisions else 0.0,
        "unsupported_query_count": len(unsupported_decisions),
        "note": "Offline proxy: answers are represented by retrieved passages; this does not judge generated answer entailment.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense and hybrid document retrieval.")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("dataset.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    chunks = dataset["chunks"]
    texts = [chunk["text"] for chunk in chunks]
    questions = [item["question"] for item in dataset["questions"]]
    unsupported_questions = [item["question"] for item in dataset.get("unsupported_questions", [])]
    all_queries = questions + unsupported_questions
    pipeline = DocumentPipeline()
    vectors = pipeline.embed_texts(texts).astype(np.float32)
    query_vectors = pipeline.embed_texts(all_queries).astype(np.float32)
    faiss.normalize_L2(vectors)
    faiss.normalize_L2(query_vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    dense_scores, dense_positions = index.search(query_vectors, len(chunks))

    dense_rankings: dict[str, list[str]] = {}
    hybrid_rankings: dict[str, list[str]] = {}
    all_dense_scores: dict[str, dict[str, float]] = {}
    for row, question in enumerate(all_queries):
        dense_order = [int(position) for position in dense_positions[row]]
        ranked_ids = [chunks[position]["id"] for position in dense_order]
        all_dense_scores[question] = {
            chunks[position]["id"]: float(dense_scores[row][rank])
            for rank, position in enumerate(dense_order)
        }
        if question not in questions:
            continue
        dense_rankings[question] = ranked_ids
        keyword_scores = DocumentService._bm25_scores(question, texts)
        keyword_order = sorted(range(len(chunks)), key=keyword_scores.__getitem__, reverse=True)
        dense_ranks = {position: rank for rank, position in enumerate(dense_order, start=1)}
        keyword_ranks = {position: rank for rank, position in enumerate(keyword_order, start=1)}
        rrf_order = sorted(
            range(len(chunks)),
            key=lambda position: 1 / (60 + dense_ranks[position]) + 1 / (60 + keyword_ranks[position]),
            reverse=True,
        )
        hybrid_rankings[question] = [chunks[position]["id"] for position in rrf_order]

    ks = (1, 3, 5)
    result = {
        "dataset": str(args.dataset.name),
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_model_revision": settings.EMBEDDING_MODEL_REVISION,
        "hybrid_fusion": "Reciprocal Rank Fusion (k=60) of dense cosine-similarity rank and BM25 rank",
        "limitations": [
            "Small synthetic dataset; results are not representative of arbitrary PDFs.",
            "Relevant chunk labels were manually checked against this dataset only.",
            "No reranker or generated-answer quality is evaluated here.",
            "Embedding model inference requires its weights to be available locally or downloadable.",
        ],
        "dense": evaluate_rankings(dataset, dense_rankings, ks),
        "hybrid": evaluate_rankings(dataset, hybrid_rankings, ks),
        "offline_answer_grounding": {
            "dense": evaluate_grounding(dataset, dense_rankings, all_dense_scores),
            "hybrid": evaluate_grounding(dataset, hybrid_rankings, all_dense_scores),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
