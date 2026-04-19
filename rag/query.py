"""Run a hybrid search + rerank pipeline against Qdrant."""
from __future__ import annotations

import os
import sys

from FlagEmbedding import FlagReranker
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models

COLLECTION = os.getenv("QDRANT_COLLECTION", "kb")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY = os.getenv("QDRANT_API_KEY", "")
TENANT_ID = os.getenv("TENANT_ID", "default")


def run(query: str, top_k: int = 6) -> list[dict]:
    qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
    dense = next(TextEmbedding("BAAI/bge-m3").embed([query]))
    sparse = next(SparseTextEmbedding("Qdrant/bm25").embed([query]))

    hybrid = qc.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dense.tolist(), using="dense", limit=40),
            models.Prefetch(query=sparse.as_object(), using="bm25", limit=40),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=models.Filter(must=[
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=TENANT_ID))
        ]),
        limit=40,
    ).points

    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    pairs = [[query, p.payload["text"]] for p in hybrid]
    scores = reranker.compute_score(pairs, normalize=True)
    ranked = sorted(zip(hybrid, scores), key=lambda x: -x[1])
    return [{"text": p.payload["text"], "score": s, "source": p.payload.get("source")} for p, s in ranked[:top_k]]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "¿cuál es el período de devolución?"
    for hit in run(q):
        print(f"[{hit['score']:.3f}] {hit['text'][:180]}...")
